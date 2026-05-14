import time

import pytest
from fastapi import HTTPException

from nntile_gateway.auth import (
    AdminAuth,
    ApiKeyAuth,
    generate_key,
    hash_key,
)
from nntile_gateway.storage.base import KeyRecord
from nntile_gateway.storage.memory import InMemoryStorage


def test_generate_key_is_random_and_hashes_stable():
    p1, h1 = generate_key()
    p2, h2 = generate_key()
    assert p1 != p2
    assert h1 == hash_key(p1)
    assert h2 == hash_key(p2)
    assert p1.startswith("nnt_")


def test_admin_auth_accepts_correct_and_rejects_wrong():
    guard = AdminAuth("secret")
    guard(authorization="Bearer secret")  # no raise
    with pytest.raises(HTTPException) as ei:
        guard(authorization=None)
    assert ei.value.status_code == 401
    with pytest.raises(HTTPException) as ei:
        guard(authorization="Bearer wrong")
    assert ei.value.status_code == 403
    with pytest.raises(HTTPException) as ei:
        guard(authorization="Basic secret")
    assert ei.value.status_code == 401


def test_api_key_auth_caches_positive_lookups():
    storage = InMemoryStorage()
    plaintext, key_hash = generate_key()
    rec = KeyRecord(
        id="k1", name="n", key_hash=key_hash, created_at=time.time())
    storage.add_key(rec)

    auth = ApiKeyAuth(storage, ttl_seconds=60, cache_size=8)
    out1 = auth(authorization=f"Bearer {plaintext}")
    assert out1.id == "k1"

    # Wipe storage; cached entry should still resolve.
    storage._keys.clear()
    storage._key_by_hash.clear()
    out2 = auth(authorization=f"Bearer {plaintext}")
    assert out2.id == "k1"


def test_api_key_auth_invalidation_clears_cache():
    storage = InMemoryStorage()
    plaintext, key_hash = generate_key()
    rec = KeyRecord(
        id="k1", name="n", key_hash=key_hash, created_at=time.time())
    storage.add_key(rec)

    auth = ApiKeyAuth(storage, ttl_seconds=60, cache_size=8)
    auth(authorization=f"Bearer {plaintext}")  # warm cache
    storage.revoke_key("k1")
    auth.invalidate(key_hash)

    with pytest.raises(HTTPException) as ei:
        auth(authorization=f"Bearer {plaintext}")
    assert ei.value.status_code == 401


def test_api_key_auth_rejects_expired():
    storage = InMemoryStorage()
    plaintext, key_hash = generate_key()
    storage.add_key(KeyRecord(
        id="k1", name="n", key_hash=key_hash,
        created_at=time.time() - 100, expires_at=time.time() - 1))

    auth = ApiKeyAuth(storage, ttl_seconds=60, cache_size=8)
    with pytest.raises(HTTPException) as ei:
        auth(authorization=f"Bearer {plaintext}")
    assert ei.value.status_code == 401
