import threading
import time

from nntile_gateway.storage.base import KeyRecord, ModelRecord


class InMemoryStorage:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, ModelRecord] = {}
        self._keys: dict[str, KeyRecord] = {}
        self._key_by_hash: dict[str, str] = {}

    def add_model(self, record: ModelRecord) -> None:
        with self._lock:
            if record.spec.id in self._models:
                raise ValueError(
                    f"model {record.spec.id!r} already registered")
            self._models[record.spec.id] = record

    def remove_model(self, model_id: str) -> bool:
        with self._lock:
            return self._models.pop(model_id, None) is not None

    def get_model(self, model_id: str) -> ModelRecord | None:
        with self._lock:
            return self._models.get(model_id)

    def list_models(self) -> list[ModelRecord]:
        with self._lock:
            return list(self._models.values())

    def add_key(self, record: KeyRecord) -> None:
        with self._lock:
            self._keys[record.id] = record
            self._key_by_hash[record.key_hash] = record.id

    def revoke_key(self, key_id: str) -> bool:
        with self._lock:
            record = self._keys.get(key_id)
            if record is None or record.revoked_at is not None:
                return False
            record.revoked_at = time.time()
            return True

    def get_key_by_hash(self, key_hash: str) -> KeyRecord | None:
        with self._lock:
            kid = self._key_by_hash.get(key_hash)
            if kid is None:
                return None
            return self._keys.get(kid)

    def list_keys(self) -> list[KeyRecord]:
        with self._lock:
            return list(self._keys.values())
