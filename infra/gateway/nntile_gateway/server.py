import threading
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, status

from nntile_gateway.auth import (
    AdminAuth,
    ApiKeyAuth,
    generate_key,
)
from nntile_gateway.config import GatewayConfig
from nntile_gateway.engine import GenerateOptions
from nntile_gateway.model_loader import ModelLoader, NNTileModelLoader
from nntile_gateway.model_registry import ModelRegistry
from nntile_gateway.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
    CreateKeyRequest,
    CreateKeyResponse,
    EmbeddingObject,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    FillMaskCandidate,
    FillMaskRequest,
    FillMaskResponse,
    FillMaskUsage,
    KeyInfo,
    ModelInfo,
    ModelSpec,
    OpenAIModelList,
    OpenAIModelObject,
)
from nntile_gateway.storage.base import KeyRecord, Storage
from nntile_gateway.storage.memory import InMemoryStorage
from nntile_gateway.storage.sqlite import SqliteStorage


def _default_storage(config: GatewayConfig) -> Storage:
    if config.storage == "sqlite":
        return SqliteStorage(config.sqlite_path)
    return InMemoryStorage()


def build_app(
    config: GatewayConfig,
    storage: Storage | None = None,
    loader: ModelLoader | None = None,
) -> FastAPI:
    config.validate()
    storage = storage if storage is not None else _default_storage(config)
    loader = loader or NNTileModelLoader()
    registry = ModelRegistry(storage, loader)
    registry.rehydrate_from_storage()

    require_admin = AdminAuth(config.admin_token)
    require_key = ApiKeyAuth(
        storage,
        ttl_seconds=config.auth_cache_ttl,
        cache_size=config.auth_cache_size,
    )

    generate_lock = threading.Lock()

    app = FastAPI(title="nntile-gateway")

    # --- admin: models ---------------------------------------------------

    @app.post(
        "/admin/models",
        response_model=ModelInfo,
        dependencies=[Depends(require_admin)],
    )
    def admin_add_model(spec: ModelSpec) -> ModelInfo:
        try:
            loaded = registry.register(spec)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"failed to load model: {type(exc).__name__}: {exc}",
            )
        return registry.to_info(loaded)

    @app.delete(
        "/admin/models/{model_id}",
        dependencies=[Depends(require_admin)],
    )
    def admin_remove_model(model_id: str) -> dict[str, bool]:
        removed = registry.unregister(model_id)
        if not removed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"model {model_id!r} not found",
            )
        return {"removed": True}

    @app.get(
        "/admin/models",
        response_model=list[ModelInfo],
        dependencies=[Depends(require_admin)],
    )
    def admin_list_models() -> list[ModelInfo]:
        return [registry.to_info(m) for m in registry.list()]

    # --- admin: keys -----------------------------------------------------

    @app.post(
        "/admin/keys",
        response_model=CreateKeyResponse,
        dependencies=[Depends(require_admin)],
    )
    def admin_create_key(req: CreateKeyRequest) -> CreateKeyResponse:
        plaintext, key_hash = generate_key()
        record = KeyRecord(
            id=str(uuid.uuid4()),
            name=req.name,
            key_hash=key_hash,
            created_at=time.time(),
            expires_at=req.expires_at,
        )
        storage.add_key(record)
        return CreateKeyResponse(id=record.id, name=record.name, key=plaintext)

    @app.delete(
        "/admin/keys/{key_id}",
        dependencies=[Depends(require_admin)],
    )
    def admin_revoke_key(key_id: str) -> dict[str, bool]:
        revoked = storage.revoke_key(key_id)
        if not revoked:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"key {key_id!r} not found or already revoked",
            )
        for k in storage.list_keys():
            if k.id == key_id:
                require_key.invalidate(k.key_hash)
                break
        return {"revoked": True}

    @app.get(
        "/admin/keys",
        response_model=list[KeyInfo],
        dependencies=[Depends(require_admin)],
    )
    def admin_list_keys() -> list[KeyInfo]:
        return [
            KeyInfo(
                id=k.id,
                name=k.name,
                created_at=k.created_at,
                expires_at=k.expires_at,
                revoked_at=k.revoked_at,
            )
            for k in storage.list_keys()
        ]

    # --- inference: OpenAI-compatible ------------------------------------

    def _ready_model(model_id: str):
        loaded = registry.get(model_id)
        if loaded is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"model {model_id!r} not found",
            )
        if loaded.status != "ready" or loaded.engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"model {model_id!r} not ready "
                    f"(status={loaded.status}, error={loaded.error})"
                ),
            )
        return loaded

    @app.get(
        "/v1/models",
        response_model=OpenAIModelList,
        dependencies=[Depends(require_key)],
    )
    def v1_list_models() -> OpenAIModelList:
        items = []
        for loaded in registry.list():
            if loaded.status != "ready":
                continue
            items.append(OpenAIModelObject(
                id=loaded.spec.id,
                created=int(loaded.created_at),
            ))
        return OpenAIModelList(data=items)

    def _require_generation_engine(loaded, model_id: str):
        engine = loaded.engine
        if not hasattr(engine, "generate"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"model {model_id!r} (family={loaded.spec.family!r}) "
                    "does not support text generation"
                ),
            )
        return engine

    @app.post(
        "/v1/completions",
        response_model=CompletionResponse,
        dependencies=[Depends(require_key)],
    )
    def v1_completions(req: CompletionRequest) -> CompletionResponse:
        loaded = _ready_model(req.model)
        engine = _require_generation_engine(loaded, req.model)
        opts = GenerateOptions(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        with generate_lock:
            result = engine.generate(req.prompt, opts)
        return CompletionResponse(
            id="cmpl-" + uuid.uuid4().hex,
            created=int(time.time()),
            model=req.model,
            choices=[CompletionChoice(
                text=result.text, finish_reason=result.finish_reason)],
            usage=CompletionUsage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
        )

    @app.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        dependencies=[Depends(require_key)],
    )
    def v1_chat_completions(
        req: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        loaded = _ready_model(req.model)
        engine = _require_generation_engine(loaded, req.model)
        messages = [m.model_dump() for m in req.messages]
        prompt = engine.apply_chat_template(messages)
        opts = GenerateOptions(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        with generate_lock:
            result = engine.generate(prompt, opts)
        return ChatCompletionResponse(
            id="chatcmpl-" + uuid.uuid4().hex,
            created=int(time.time()),
            model=req.model,
            choices=[ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=result.text),
                finish_reason=result.finish_reason,
            )],
            usage=CompletionUsage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
        )

    @app.post(
        "/v1/fill_mask",
        response_model=FillMaskResponse,
        dependencies=[Depends(require_key)],
    )
    def v1_fill_mask(req: FillMaskRequest) -> FillMaskResponse:
        loaded = _ready_model(req.model)
        engine = loaded.engine
        if not hasattr(engine, "fill_mask"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"model {req.model!r} (family={loaded.spec.family!r}, "
                    f"task={loaded.spec.task!r}) does not support "
                    "/v1/fill_mask"
                ),
            )
        with generate_lock:
            try:
                result = engine.fill_mask(  # type: ignore[union-attr]
                    req.input, req.top_k)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                )
        data = [
            [FillMaskCandidate(
                token=c.token, token_str=c.token_str,
                score=c.score, sequence=c.sequence)
             for c in per_mask]
            for per_mask in result.candidates
        ]
        return FillMaskResponse(
            model=req.model,
            data=data,
            usage=FillMaskUsage(
                prompt_tokens=result.prompt_tokens,
                total_tokens=result.prompt_tokens,
            ),
        )

    @app.post(
        "/v1/embeddings",
        response_model=EmbeddingsResponse,
        dependencies=[Depends(require_key)],
    )
    def v1_embeddings(req: EmbeddingsRequest) -> EmbeddingsResponse:
        loaded = _ready_model(req.model)
        engine = loaded.engine
        if not hasattr(engine, "embed"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"model {req.model!r} (family={loaded.spec.family!r}) "
                    "does not support /v1/embeddings"
                ),
            )
        inputs = req.input if isinstance(req.input, list) else [req.input]
        data: list[EmbeddingObject] = []
        total_tokens = 0
        for idx, text in enumerate(inputs):
            with generate_lock:
                res = engine.embed(text)  # type: ignore[union-attr]
            data.append(EmbeddingObject(
                embedding=res.embedding, index=idx))
            total_tokens += res.prompt_tokens
        return EmbeddingsResponse(
            model=req.model,
            data=data,
            usage=EmbeddingsUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
            ),
        )

    # --- meta ------------------------------------------------------------

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
