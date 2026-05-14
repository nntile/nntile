import json
import sqlite3
import threading

from nntile_gateway.schemas import ModelSpec
from nntile_gateway.storage.base import KeyRecord, ModelRecord


_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    expires_at REAL,
    revoked_at REAL
);

CREATE INDEX IF NOT EXISTS api_keys_hash_idx ON api_keys(key_hash);
"""


class SqliteStorage:
    """SQLite-backed Storage with the same surface as InMemoryStorage.

    Uses a single connection with check_same_thread=False, guarded by a
    module-level lock so admin and inference paths can share it.
    """

    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- models ---------------------------------------------------------

    def add_model(self, record: ModelRecord) -> None:
        payload = record.spec.model_dump_json()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO models (id, spec_json, created_at) "
                    "VALUES (?, ?, ?)",
                    (record.spec.id, payload, record.created_at),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"model {record.spec.id!r} already registered") from exc

    def remove_model(self, model_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM models WHERE id = ?", (model_id,))
            return cur.rowcount > 0

    def get_model(self, model_id: str) -> ModelRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT spec_json, created_at FROM models WHERE id = ?",
                (model_id,),
            ).fetchone()
        if row is None:
            return None
        return ModelRecord(
            spec=ModelSpec.model_validate_json(row["spec_json"]),
            created_at=row["created_at"],
        )

    def list_models(self) -> list[ModelRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT spec_json, created_at FROM models "
                "ORDER BY created_at"
            ).fetchall()
        return [
            ModelRecord(
                spec=ModelSpec.model_validate_json(r["spec_json"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # --- keys -----------------------------------------------------------

    def add_key(self, record: KeyRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO api_keys "
                "(id, name, key_hash, created_at, expires_at, revoked_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.id, record.name, record.key_hash,
                    record.created_at, record.expires_at, record.revoked_at,
                ),
            )

    def revoke_key(self, key_id: str) -> bool:
        import time

        with self._lock:
            cur = self._conn.execute(
                "UPDATE api_keys SET revoked_at = ? "
                "WHERE id = ? AND revoked_at IS NULL",
                (time.time(), key_id),
            )
            return cur.rowcount > 0

    def get_key_by_hash(self, key_hash: str) -> KeyRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, key_hash, created_at, expires_at, "
                "revoked_at FROM api_keys WHERE key_hash = ?",
                (key_hash,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_key(row)

    def list_keys(self) -> list[KeyRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, key_hash, created_at, expires_at, "
                "revoked_at FROM api_keys ORDER BY created_at"
            ).fetchall()
        return [_row_to_key(r) for r in rows]


def _row_to_key(row: sqlite3.Row) -> KeyRecord:
    return KeyRecord(
        id=row["id"],
        name=row["name"],
        key_hash=row["key_hash"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        revoked_at=row["revoked_at"],
    )
