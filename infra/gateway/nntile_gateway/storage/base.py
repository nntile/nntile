from dataclasses import dataclass
from typing import Protocol

from nntile_gateway.schemas import ModelSpec


@dataclass
class ModelRecord:
    spec: ModelSpec
    created_at: float


@dataclass
class KeyRecord:
    id: str
    name: str
    key_hash: str
    created_at: float
    expires_at: float | None = None
    revoked_at: float | None = None


class Storage(Protocol):
    def add_model(self, record: ModelRecord) -> None: ...
    def remove_model(self, model_id: str) -> bool: ...
    def get_model(self, model_id: str) -> ModelRecord | None: ...
    def list_models(self) -> list[ModelRecord]: ...

    def add_key(self, record: KeyRecord) -> None: ...
    def revoke_key(self, key_id: str) -> bool: ...
    def get_key_by_hash(self, key_hash: str) -> KeyRecord | None: ...
    def list_keys(self) -> list[KeyRecord]: ...
