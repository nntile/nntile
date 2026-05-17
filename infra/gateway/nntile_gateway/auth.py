"""FastAPI dependencies for admin- and user-key authentication.

`AdminAuth` gates `/admin/*` with a single shared bearer token from
the gateway config; `ApiKeyAuth` gates `/v1/*` with per-key tokens
issued via `/admin/keys` and stored hashed. Both raise an HTTPException
on failure so they slot directly into `Depends(...)` on a route."""

import hashlib
import hmac
import secrets
import time

from cachetools import TTLCache
from fastapi import Header, HTTPException, status
from nntile_gateway.storage.base import KeyRecord, Storage


def hash_key(plaintext: str) -> str:
    """Stable hex SHA-256 of a plaintext key, used for storage lookup.

    The plaintext is shown to the admin only once at issuance; we keep
    only the hash so a database leak doesn't yield usable keys."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def generate_key() -> tuple[str, str]:
    """Mint a fresh API key. Returns `(plaintext, hash)`.

    Plaintext is `nnt_` + 32 url-safe bytes (~256 bits of entropy);
    only the hash is persisted. The caller must return the plaintext
    to the user once and never again."""
    plaintext = "nnt_" + secrets.token_urlsafe(32)
    return plaintext, hash_key(plaintext)


def _extract_bearer(authorization: str | None) -> str:
    """Parse a `Bearer <token>` header value.

    Raises 401 on missing/malformed headers (with WWW-Authenticate so
    OpenAI clients re-prompt cleanly). Returns the raw token only;
    does not validate it."""
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
    """Dependency that gates admin routes with a shared bearer token.

    Uses `hmac.compare_digest` for constant-time comparison so a
    timing oracle can't recover the admin token byte by byte."""

    def __init__(self, admin_token: str) -> None:
        self._admin_token = admin_token

    def __call__(
        self, authorization: str | None = Header(default=None),
    ) -> None:
        token = _extract_bearer(authorization)
        if not hmac.compare_digest(token, self._admin_token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid admin token",
            )


_NEGATIVE = object()


class ApiKeyAuth:
    """Dependency that gates user routes with per-key bearer tokens.

    Looks up the key's hash in `Storage`, with a positive-result
    in-process TTLCache to keep hot-path latency low. Negative
    lookups (unknown keys) are not cached, so a key issued via
    `/admin/keys` is usable on its first request.

    Returns the matched `KeyRecord` on success; the FastAPI dependency
    system makes that record available to route handlers via the
    standard `Depends(...)` typed parameter."""

    def __init__(
        self,
        storage: Storage,
        ttl_seconds: int,
        cache_size: int,
    ) -> None:
        self._storage = storage
        self._cache: TTLCache = TTLCache(maxsize=cache_size, ttl=ttl_seconds)

    def invalidate(self, key_hash: str) -> None:
        """Drop a cached positive lookup. Called after revocation so a
        revoked key stops authenticating immediately rather than at
        the next TTL expiry."""
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
