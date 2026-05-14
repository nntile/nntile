import threading
import time
from dataclasses import dataclass
from typing import Literal

from typing import Any

from nntile_gateway.model_loader import ModelLoader
from nntile_gateway.schemas import ModelInfo, ModelSpec
from nntile_gateway.storage.base import ModelRecord, Storage


@dataclass
class LoadedModel:
    spec: ModelSpec
    status: Literal["loading", "ready", "error"]
    error: str | None = None
    # GatewayEngine (causal/seq2seq) or EmbeddingEngine (encoder-only).
    # Endpoints route by hasattr(engine, "generate" | "embed").
    engine: Any = None
    created_at: float = 0.0


class ModelRegistry:
    """Thread-safe registry of materialized models.

    Loads happen sequentially under a single lock because the shared
    nntile.Context is process-wide and concurrent materialization is unsafe.
    """

    def __init__(self, storage: Storage, loader: ModelLoader) -> None:
        self._storage = storage
        self._loader = loader
        self._models: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()

    def register(self, spec: ModelSpec) -> LoadedModel:
        with self._lock:
            if spec.id in self._models:
                raise ValueError(f"model {spec.id!r} already registered")
            now = time.time()
            loaded = self._load_locked(spec, created_at=now)
            try:
                self._storage.add_model(
                    ModelRecord(spec=spec, created_at=now))
            except Exception:
                # Roll back the in-memory entry so the next attempt can retry.
                self._models.pop(spec.id, None)
                raise
            return loaded

    def rehydrate_from_storage(self) -> list[LoadedModel]:
        """Load every persisted model into memory.

        Failures are recorded on the LoadedModel (status='error') rather
        than raised, so a bad model doesn't prevent the server from
        starting and the admin can drop it via DELETE.
        """
        results: list[LoadedModel] = []
        with self._lock:
            for record in self._storage.list_models():
                if record.spec.id in self._models:
                    continue
                results.append(self._load_locked(
                    record.spec,
                    created_at=record.created_at,
                    swallow_errors=True,
                ))
        return results

    def _load_locked(
        self,
        spec: ModelSpec,
        created_at: float,
        swallow_errors: bool = False,
    ) -> LoadedModel:
        loaded = LoadedModel(
            spec=spec, status="loading", created_at=created_at)
        self._models[spec.id] = loaded
        try:
            loaded.engine = self._loader.load(spec)
            loaded.status = "ready"
        except Exception as exc:  # noqa: BLE001
            loaded.status = "error"
            loaded.error = f"{type(exc).__name__}: {exc}"
            if not swallow_errors:
                raise
        return loaded

    def unregister(self, model_id: str) -> bool:
        with self._lock:
            removed = self._models.pop(model_id, None) is not None
            self._storage.remove_model(model_id)
            return removed

    def get(self, model_id: str) -> LoadedModel | None:
        with self._lock:
            return self._models.get(model_id)

    def list(self) -> list[LoadedModel]:
        with self._lock:
            return list(self._models.values())

    def to_info(self, loaded: LoadedModel) -> ModelInfo:
        return ModelInfo(
            id=loaded.spec.id,
            family=loaded.spec.family,
            hf_name=loaded.spec.hf_name,
            dtype=loaded.spec.dtype,
            max_seq_len=loaded.spec.max_seq_len,
            status=loaded.status,
            error=loaded.error,
        )
