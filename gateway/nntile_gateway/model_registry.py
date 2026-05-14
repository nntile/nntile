import threading
import time
from dataclasses import dataclass
from typing import Literal

from nntile_gateway.engine import GatewayEngine
from nntile_gateway.model_loader import ModelLoader
from nntile_gateway.schemas import ModelInfo, ModelSpec
from nntile_gateway.storage.base import ModelRecord, Storage


@dataclass
class LoadedModel:
    spec: ModelSpec
    status: Literal["loading", "ready", "error"]
    error: str | None = None
    engine: GatewayEngine | None = None
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
            loaded = LoadedModel(
                spec=spec, status="loading", created_at=now)
            self._models[spec.id] = loaded
            try:
                loaded.engine = self._loader.load(spec)
                loaded.status = "ready"
                self._storage.add_model(
                    ModelRecord(spec=spec, created_at=now))
            except Exception as exc:  # noqa: BLE001
                loaded.status = "error"
                loaded.error = f"{type(exc).__name__}: {exc}"
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
