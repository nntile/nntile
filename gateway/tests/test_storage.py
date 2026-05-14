import time

from nntile_gateway.schemas import ModelSpec
from nntile_gateway.storage.base import KeyRecord, ModelRecord
from nntile_gateway.storage.memory import InMemoryStorage


def _spec(model_id: str = "m1") -> ModelSpec:
    return ModelSpec(id=model_id, family="llama", hf_name="hf/m1")


def test_model_add_get_list_remove():
    s = InMemoryStorage()
    rec = ModelRecord(spec=_spec("m1"), created_at=time.time())
    s.add_model(rec)
    assert s.get_model("m1") is rec
    assert [r.spec.id for r in s.list_models()] == ["m1"]
    assert s.remove_model("m1") is True
    assert s.remove_model("m1") is False
    assert s.get_model("m1") is None


def test_model_duplicate_rejected():
    s = InMemoryStorage()
    s.add_model(ModelRecord(spec=_spec("m1"), created_at=time.time()))
    try:
        s.add_model(ModelRecord(spec=_spec("m1"), created_at=time.time()))
    except ValueError:
        return
    raise AssertionError("expected duplicate to raise")


def test_key_add_lookup_revoke():
    s = InMemoryStorage()
    rec = KeyRecord(
        id="k1", name="alice", key_hash="hash-1", created_at=time.time())
    s.add_key(rec)
    assert s.get_key_by_hash("hash-1") is rec
    assert s.get_key_by_hash("nope") is None
    assert [k.id for k in s.list_keys()] == ["k1"]
    assert s.revoke_key("k1") is True
    assert s.revoke_key("k1") is False
    assert s.get_key_by_hash("hash-1").revoked_at is not None
