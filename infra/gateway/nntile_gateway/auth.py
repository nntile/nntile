import hashlib
import hmac
import secrets
import time

from cachetools import TTLCache
from fastapi import Header, HTTPException, status

from nntile_gateway.storage.base import KeyRecord, Storage


def hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def generate_key() -> tuple[str, str]:
    plaintext = "nnt_" + secrets.token_urlsafe(32)
    return plaintext, hash_key(plaintext)


def _extract_bearer(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    parts = authorization.split(maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization must be 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return parts[1].strip()


class AdminAuth:
    def __init__(self, admin_token: str) -> None:
        self._admin_token = admin_token

    def __call__(self, authorization: str | None = Header(default=None)) -> None:
        token = _extract_bearer(authorization)
        if not hmac.compare_digest(token, self._admin_token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid admin token",
            )


_NEGATIVE = object()


class ApiKeyAuth:
    def __init__(
        self,
        storage: Storage,
        ttl_seconds: int,
        cache_size: int,
    ) -> None:
        self._storage = storage
        self._cache: TTLCache = TTLCache(maxsize=cache_size, ttl=ttl_seconds)

    def invalidate(self, key_hash: str) -> None:
        self._cache.pop(key_hash, None)

    def __call__(
        self, authorization: str | None = Header(default=None)
    ) -> KeyRecord:
        token = _extract_bearer(authorization)
        key_hash = hash_key(token)
        cached = self._cache.get(key_hash, _NEGATIVE)
        if cached is _NEGATIVE:
            record = self._storage.get_key_by_hash(key_hash)
            self._cache[key_hash] = record
        else:
            record = cached  # type: ignore[assignment]
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unknown API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if record.revoked_at is not None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key revoked",
            )
        if record.expires_at is not None and record.expires_at < time.time():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key expired",
            )
        return record
