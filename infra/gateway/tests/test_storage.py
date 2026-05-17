"""Storage-backend contract test, run against both
`InMemoryStorage` and `SqliteStorage` via parametrization.
Plus a persistence check that `SqliteStorage` round-trips
across reopens."""

import time

import pytest
from nntile_gateway.schemas import ModelSpec
from nntile_gateway.storage.base import KeyRecord, ModelRecord
from nntile_gateway.storage.memory import InMemoryStorage
from nntile_gateway.storage.sqlite import SqliteStorage


def _spec(model_id: str = "m1") -> ModelSpec:
    return ModelSpec(id=model_id, family="llama", hf_name="hf/m1")


@pytest.fixture(params=["memory", "sqlite"])
def storage_impl(request, tmp_path):
    if request.param == "memory":
        yield InMemoryStorage()
    else:
        s = SqliteStorage(str(tmp_path / "gateway.sqlite3"))
        try:
            yield s
        finally:
            s.close()


def test_model_add_get_list_remove(storage_impl):
    s = storage_impl
    rec = ModelRecord(spec=_spec("m1"), created_at=time.time())
    s.add_model(rec)
    got = s.get_model("m1")
    assert got is not None and got.spec.id == "m1"
    assert [r.spec.id for r in s.list_models()] == ["m1"]
    assert s.remove_model("m1") is True
    assert s.remove_model("m1") is False
    assert s.get_model("m1") is None


def test_model_duplicate_rejected(storage_impl):
    s = storage_impl
    s.add_model(ModelRecord(spec=_spec("m1"), created_at=time.time()))
    with pytest.raises(ValueError):
        s.add_model(ModelRecord(spec=_spec("m1"), created_at=time.time()))


def test_key_add_lookup_revoke(storage_impl):
    s = storage_impl
    rec = KeyRecord(
        id="k1", name="alice", key_hash="hash-1", created_at=time.time())
    s.add_key(rec)
    got = s.get_key_by_hash("hash-1")
    assert got is not None and got.id == "k1"
    assert s.get_key_by_hash("nope") is None
    assert [k.id for k in s.list_keys()] == ["k1"]
    assert s.revoke_key("k1") is True
    assert s.revoke_key("k1") is False
    revoked = s.get_key_by_hash("hash-1")
    assert revoked is not None and revoked.revoked_at is not None


def test_sqlite_persists_across_reopens(tmp_path):
    path = str(tmp_path / "g.sqlite3")
    s = SqliteStorage(path)
    s.add_model(ModelRecord(spec=_spec("m1"), created_at=time.time()))
    s.add_key(KeyRecord(
        id="k1", name="alice", key_hash="h1", created_at=time.time()))
    s.close()

    s2 = SqliteStorage(path)
    assert [r.spec.id for r in s2.list_models()] == ["m1"]
    assert [k.id for k in s2.list_keys()] == ["k1"]
    s2.close()
