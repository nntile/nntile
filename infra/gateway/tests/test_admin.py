from tests.conftest import admin_headers

MODEL_PAYLOAD = {
    "id": "llama-test",
    "family": "llama",
    "hf_name": "stub/llama-test",
    "dtype": "fp32",
    "max_seq_len": 32,
}


def test_admin_endpoints_require_admin_token(client):
    r = client.get("/admin/models")
    assert r.status_code == 401
    r = client.get(
        "/admin/models", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 403


def test_register_list_remove_model(client, loader):
    r = client.post(
        "/admin/models", json=MODEL_PAYLOAD, headers=admin_headers())
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["id"] == "llama-test"
    assert info["status"] == "ready"
    assert "llama-test" in loader.engines

    r = client.get("/admin/models", headers=admin_headers())
    assert r.status_code == 200
    assert [m["id"] for m in r.json()] == ["llama-test"]

    r = client.post(
        "/admin/models", json=MODEL_PAYLOAD, headers=admin_headers())
    assert r.status_code == 409

    r = client.delete(
        "/admin/models/llama-test", headers=admin_headers())
    assert r.status_code == 200
    assert client.get(
        "/admin/models", headers=admin_headers()).json() == []


def test_register_model_load_failure_returns_500(client, loader):
    loader.fail_on.add("broken")
    payload = {**MODEL_PAYLOAD, "id": "broken"}
    r = client.post(
        "/admin/models", json=payload, headers=admin_headers())
    assert r.status_code == 500
    assert "forced load failure" in r.text


def test_create_list_revoke_key(client):
    r = client.post(
        "/admin/keys",
        json={"name": "alice"},
        headers=admin_headers(),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["key"].startswith("nnt_")
    key_id = body["id"]

    r = client.get("/admin/keys", headers=admin_headers())
    assert r.status_code == 200
    keys = r.json()
    assert len(keys) == 1
    assert keys[0]["id"] == key_id
    assert "key" not in keys[0]

    r = client.delete(
        f"/admin/keys/{key_id}", headers=admin_headers())
    assert r.status_code == 200
    keys = client.get("/admin/keys", headers=admin_headers()).json()
    assert keys[0]["revoked_at"] is not None

    r = client.delete(
        f"/admin/keys/{key_id}", headers=admin_headers())
    assert r.status_code == 404
