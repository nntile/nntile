"""OpenAI-compatible /v1/* routes: model listing, completions
and chat-completions happy paths, error paths (unknown model,
missing/revoked key), and the cache-invalidation hook after
revoking a key."""

from tests.conftest import admin_headers

MODEL_PAYLOAD = {
    "id": "llama-test",
    "family": "llama",
    "hf_name": "stub/llama-test",
    "dtype": "fp32",
    "max_seq_len": 32,
}


def _register_model(client) -> None:
    r = client.post(
        "/admin/models", json=MODEL_PAYLOAD, headers=admin_headers())
    assert r.status_code == 200, r.text


def _issue_key(client, name: str = "alice") -> str:
    r = client.post(
        "/admin/keys", json={"name": name}, headers=admin_headers())
    assert r.status_code == 200, r.text
    return r.json()["key"]


def _user_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_v1_requires_api_key(client):
    _register_model(client)
    r = client.get("/v1/models")
    assert r.status_code == 401
    r = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer not-a-real-key"},
    )
    assert r.status_code == 401


def test_v1_models_lists_only_ready(client):
    _register_model(client)
    key = _issue_key(client)
    r = client.get("/v1/models", headers=_user_headers(key))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["llama-test"]


def test_v1_completions(client):
    _register_model(client)
    key = _issue_key(client)
    r = client.post(
        "/v1/completions",
        headers=_user_headers(key),
        json={
            "model": "llama-test",
            "prompt": "hello world",
            "max_tokens": 4,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "llama-test"
    assert body["choices"][0]["text"] == " [echo:llama-test]"
    assert body["usage"]["completion_tokens"] == 2
    assert body["usage"]["prompt_tokens"] == 2


def test_v1_completions_unknown_model(client):
    key = _issue_key(client)
    r = client.post(
        "/v1/completions",
        headers=_user_headers(key),
        json={"model": "nope", "prompt": "x"},
    )
    assert r.status_code == 404


def test_v1_chat_completions(client):
    _register_model(client)
    key = _issue_key(client)
    r = client.post(
        "/v1/chat/completions",
        headers=_user_headers(key),
        json={
            "model": "llama-test",
            "messages": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 4,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    msg = body["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert msg["content"] == " [echo:llama-test]"


def test_revoked_key_rejected_after_invalidation(client):
    _register_model(client)
    key = _issue_key(client)
    # warm cache
    assert client.get(
        "/v1/models", headers=_user_headers(key)).status_code == 200

    keys = client.get("/admin/keys", headers=admin_headers()).json()
    kid = keys[0]["id"]
    assert client.delete(
        f"/admin/keys/{kid}", headers=admin_headers()).status_code == 200

    r = client.get("/v1/models", headers=_user_headers(key))
    assert r.status_code == 401
