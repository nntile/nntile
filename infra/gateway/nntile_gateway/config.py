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
        if not self.admin_token:
            raise RuntimeError(
                "NNTILE_ADMIN_TOKEN must be set to start the gateway")
        if self.storage not in ("memory", "sqlite"):
            raise RuntimeError(
                f"NNTILE_GATEWAY_STORAGE must be 'memory' or 'sqlite', "
                f"got {self.storage!r}")
