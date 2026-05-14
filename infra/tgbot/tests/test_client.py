from __future__ import annotations

import httpx
import pytest

from nntile_tgbot.client import GatewayError


async def test_list_models(make_client):
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json={"data": [
            {"id": "gpt2", "family": "gpt2", "status": "ready"},
            {"id": "tiny", "family": "llama", "status": "loading"},
        ]})

    client = make_client(handler)
    try:
        models = await client.list_models()
    finally:
        await client.aclose()
    assert seen["url"] == "http://test/v1/models"
    assert seen["auth"] == "Bearer nnt_test"
    assert [m.id for m in models] == ["gpt2", "tiny"]
    assert models[1].status == "loading"


async def test_chat_completion(make_client):
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "model": "gpt2",
            "choices": [{"index": 0, "message": {
                "role": "assistant", "content": "hello back"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2,
                      "total_tokens": 3},
        })

    client = make_client(handler)
    try:
        text = await client.chat_completion(
            model="gpt2",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
        )
    finally:
        await client.aclose()
    assert text == "hello back"
    assert seen["url"] == "http://test/v1/chat/completions"
    assert seen["body"]["model"] == "gpt2"
    assert seen["body"]["max_tokens"] == 16
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]


async def test_raises_on_non_2xx(make_client):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")
    client = make_client(handler)
    try:
        with pytest.raises(GatewayError) as ei:
            await client.list_models()
    finally:
        await client.aclose()
    assert ei.value.status_code == 401
    assert "bad key" in ei.value.body
