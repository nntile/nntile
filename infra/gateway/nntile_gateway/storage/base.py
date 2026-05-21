"""Storage protocol and record dataclasses (no I/O here).

Concrete backends live in `memory.py` and `sqlite.py`. Records carry
just the persistent bits -- the live engine object is held only in
the in-process `ModelRegistry`."""

from dataclasses import dataclass
from typing import Protocol

from nntile_gateway.schemas import ModelSpec


@dataclass
class ModelRecord:
    """One row in the `models` table: the full ModelSpec + load time."""

    spec: ModelSpec
    created_at: float


@dataclass
class KeyRecord:
    """One row in the `api_keys` table.

    Only the SHA-256 `key_hash` is stored; the plaintext is shown to
    the operator once at issuance and discarded. `revoked_at` is None
    while the key is active; `expires_at` is optional and enforced by
    `ApiKeyAuth`."""

    id: str
    name: str
    key_hash: str
    created_at: float
    expires_at: float | None = None
    revoked_at: float | None = None


class Storage(Protocol):
    """Persistence interface used by `ModelRegistry` and `ApiKeyAuth`.

    All methods are synchronous; backends are expected to internally
    serialize concurrent access (in-memory and sqlite both use a
    `threading.Lock`)."""

    def add_model(self, record: ModelRecord) -> None:
        """Persist a model spec. Raises `ValueError` on duplicate id."""

    def remove_model(self, model_id: str) -> bool:
        """Delete a model spec; returns True if it existed."""

    def get_model(self, model_id: str) -> ModelRecord | None:
        """Look up one model spec by id; None if unknown."""

    def list_models(self) -> list[ModelRecord]:
        """Every persisted model, in creation order."""

    def add_key(self, record: KeyRecord) -> None:
        """Persist an API key record (hashed; plaintext is not stored)."""

    def revoke_key(self, key_id: str) -> bool:
        """Mark a key revoked; returns True if it transitioned now."""

    def get_key_by_hash(self, key_hash: str) -> KeyRecord | None:
        """Look up by hash -- the hot path for `ApiKeyAuth`."""

    def list_keys(self) -> list[KeyRecord]:
        """Every persisted key, in creation order (admin listing)."""
