"""Gateway runtime configuration.

All knobs are read from `NNTILE_GATEWAY_*` (and `NNTILE_ADMIN_TOKEN`)
environment variables with sensible defaults so the container image
doesn't require a config file. See `infra/README.md` for the full
table."""

import os
from dataclasses import dataclass, field
from typing import Literal


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw


@dataclass
class GatewayConfig:
    """Env-driven settings, populated at construction time.

    `ncpu`/`ncuda` are forwarded to `nntile.Context` (=-1 means "use
    all"). `auth_cache_ttl`/`auth_cache_size` size the in-process
    api-key TTLCache (negative-result caching is intentionally off so
    a freshly issued key works on its first request)."""

    admin_token: str = field(default_factory=lambda: os.environ.get(
        "NNTILE_ADMIN_TOKEN", ""))
    host: str = field(default_factory=lambda: _env_str(
        "NNTILE_GATEWAY_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int(
        "NNTILE_GATEWAY_PORT", 12224))
    storage: Literal["memory", "sqlite"] = field(
        default_factory=lambda: _env_str(
            "NNTILE_GATEWAY_STORAGE", "memory"))  # type: ignore[assignment]
    sqlite_path: str = field(default_factory=lambda: _env_str(
        "NNTILE_GATEWAY_SQLITE_PATH", "gateway.sqlite3"))
    auth_cache_ttl: int = field(default_factory=lambda: _env_int(
        "NNTILE_GATEWAY_AUTH_CACHE_TTL", 60))
    auth_cache_size: int = field(default_factory=lambda: _env_int(
        "NNTILE_GATEWAY_AUTH_CACHE_SIZE", 1024))
    ncpu: int = field(default_factory=lambda: _env_int(
        "NNTILE_GATEWAY_NCPU", -1))
    ncuda: int = field(default_factory=lambda: _env_int(
        "NNTILE_GATEWAY_NCUDA", -1))

    def validate(self) -> None:
        """Raise RuntimeError if required settings are missing/invalid.

        Called by `build_app` before any storage or registry init so
        a misconfigured server fails fast at startup rather than on
        the first request."""
        if not self.admin_token:
            raise RuntimeError(
                "NNTILE_ADMIN_TOKEN must be set to start the gateway")
        if self.storage not in ("memory", "sqlite"):
            raise RuntimeError(
                f"NNTILE_GATEWAY_STORAGE must be 'memory' or 'sqlite', "
                f"got {self.storage!r}")
