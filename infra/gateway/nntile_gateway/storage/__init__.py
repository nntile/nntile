"""Persistence backends for registered models and API keys.

Two implementations behind the `Storage` protocol: `InMemoryStorage`
(fast, no I/O, contents lost on restart) and `SqliteStorage` (small
SQLite DB so `/admin/models` and `/admin/keys` survive a restart).
Selected by `NNTILE_GATEWAY_STORAGE`."""

from nntile_gateway.storage.base import KeyRecord, ModelRecord, Storage
from nntile_gateway.storage.memory import InMemoryStorage
from nntile_gateway.storage.sqlite import SqliteStorage

__all__ = [
    "KeyRecord",
    "ModelRecord",
    "Storage",
    "InMemoryStorage",
    "SqliteStorage",
]
