from fastapi.testclient import TestClient

from nntile_gateway.config import GatewayConfig
from nntile_gateway.server import build_app
from nntile_gateway.storage.sqlite import SqliteStorage
from tests.conftest import ADMIN_TOKEN, FakeLoader, admin_headers


MODEL_PAYLOAD = {
    "id": "llama-test",
    "family": "llama",
    "hf_name": "stub/llama-test",
    "dtype": "fp32",
    "max_seq_len": 32,
}


def _config(path: str) -> GatewayConfig:
    return GatewayConfig(
        admin_token=ADMIN_TOKEN,
        host="127.0.0.1",
        port=0,
        storage="sqlite",
        sqlite_path=path,
        auth_cache_ttl=60,
        auth_cache_size=64,
        ncpu=-1,
        ncuda=-1,
    )


def test_build_app_picks_sqlite_from_config(tmp_path):
    path = str(tmp_path / "g.sqlite3")
    app = build_app(_config(path), loader=FakeLoader())
    with TestClient(app) as c:
        r = c.post(
            "/admin/models", json=MODEL_PAYLOAD, headers=admin_headers())
        assert r.status_code == 200, r.text

    # Independently re-open the same sqlite file and observe the row.
    s = SqliteStorage(path)
    assert [r.spec.id for r in s.list_models()] == ["llama-test"]
    s.close()


def test_models_rehydrated_on_restart(tmp_path):
    path = str(tmp_path / "g.sqlite3")
    cfg = _config(path)

    loader1 = FakeLoader()
    app1 = build_app(cfg, loader=loader1)
    with TestClient(app1) as c:
        r = c.post(
            "/admin/models", json=MODEL_PAYLOAD, headers=admin_headers())
        assert r.status_code == 200, r.text
        kr = c.post(
            "/admin/keys", json={"name": "alice"}, headers=admin_headers())
        api_key = kr.json()["key"]

    # Second build = fresh process simulation. New loader instance proves
    # rehydration actually invoked load() again.
    loader2 = FakeLoader()
    app2 = build_app(cfg, loader=loader2)
    with TestClient(app2) as c:
        r = c.get(
            "/admin/models", headers=admin_headers())
        assert r.status_code == 200
        assert [m["id"] for m in r.json()] == ["llama-test"]
        assert "llama-test" in loader2.engines

        # Persisted key still works.
        r = c.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert r.status_code == 200
        assert [m["id"] for m in r.json()["data"]] == ["llama-test"]


def test_rehydrate_marks_failed_models_as_error_without_crashing(tmp_path):
    path = str(tmp_path / "g.sqlite3")
    cfg = _config(path)

    loader1 = FakeLoader()
    app1 = build_app(cfg, loader=loader1)
    with TestClient(app1) as c:
        assert c.post(
            "/admin/models", json=MODEL_PAYLOAD,
            headers=admin_headers()).status_code == 200

    loader2 = FakeLoader()
    loader2.fail_on.add("llama-test")
    app2 = build_app(cfg, loader=loader2)
    with TestClient(app2) as c:
        r = c.get("/admin/models", headers=admin_headers())
        assert r.status_code == 200
        info = r.json()[0]
        assert info["status"] == "error"
        assert "forced load failure" in info["error"]
