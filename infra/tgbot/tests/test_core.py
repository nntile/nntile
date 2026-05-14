from __future__ import annotations

import httpx

from nntile_tgbot.core import (
    handle_current,
    handle_models,
    handle_reset,
    handle_select,
    handle_text,
    is_authorized,
    parse_select_callback,
)
from nntile_tgbot.state import ChatStore


def _models_handler(models, status=200):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/models":
            return httpx.Response(status, json={"data": models})
        return httpx.Response(500, text="unexpected")
    return handler


def _chat_handler(reply_text):
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json
        if req.url.path == "/v1/chat/completions":
            captured["body"] = json.loads(req.content)
            return httpx.Response(200, json={
                "id": "x", "object": "chat.completion",
                "model": captured["body"]["model"],
                "choices": [{"index": 0, "message": {
                    "role": "assistant", "content": reply_text}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            })
        return httpx.Response(500, text="unexpected")
    return handler, captured


# -- authorization --------------------------------------------------

def test_is_authorized_empty_set_is_open():
    assert is_authorized(42, set()) is True
    assert is_authorized(None, set()) is True


def test_is_authorized_with_allowlist():
    assert is_authorized(42, {42}) is True
    assert is_authorized(99, {42}) is False
    assert is_authorized(None, {42}) is False


# -- /models --------------------------------------------------------

async def test_handle_models_lists_with_buttons(make_client):
    client = make_client(_models_handler([
        {"id": "gpt2", "family": "gpt2", "status": "ready"},
        {"id": "tiny", "family": "llama", "status": "ready"},
        {"id": "broken", "family": "gpt2", "status": "error"},
    ]))
    try:
        reply = await handle_models(client)
    finally:
        await client.aclose()
    assert "gpt2" in reply.text and "tiny" in reply.text
    assert "broken" in reply.text and "[error]" in reply.text
    # error model gets no button
    assert [b.label for b in reply.buttons] == ["gpt2", "tiny"]
    assert reply.buttons[0].callback_data == "select:gpt2"


async def test_handle_models_empty(make_client):
    client = make_client(_models_handler([]))
    try:
        reply = await handle_models(client)
    finally:
        await client.aclose()
    assert "No models" in reply.text
    assert reply.buttons == []


async def test_handle_models_gateway_error(make_client):
    client = make_client(_models_handler([], status=503))
    try:
        reply = await handle_models(client)
    finally:
        await client.aclose()
    assert "Gateway error" in reply.text
    assert reply.buttons == []


# -- /select --------------------------------------------------------

async def test_handle_select_success(make_client):
    client = make_client(_models_handler([
        {"id": "gpt2", "family": "gpt2", "status": "ready"},
    ]))
    store = ChatStore(history_turns=4)
    try:
        reply = await handle_select(client, store, chat_id=1, model_id="gpt2")
    finally:
        await client.aclose()
    assert "Selected model: gpt2" in reply
    assert store.get(1).selected_model == "gpt2"


async def test_handle_select_unknown(make_client):
    client = make_client(_models_handler([
        {"id": "gpt2", "family": "gpt2", "status": "ready"},
    ]))
    store = ChatStore(history_turns=4)
    try:
        reply = await handle_select(client, store, 1, "missing")
    finally:
        await client.aclose()
    assert "Unknown model" in reply
    assert store.get(1).selected_model is None


async def test_handle_select_not_ready(make_client):
    client = make_client(_models_handler([
        {"id": "gpt2", "family": "gpt2", "status": "loading"},
    ]))
    store = ChatStore(history_turns=4)
    try:
        reply = await handle_select(client, store, 1, "gpt2")
    finally:
        await client.aclose()
    assert "not ready" in reply
    assert store.get(1).selected_model is None


async def test_handle_select_empty_arg(make_client):
    client = make_client(_models_handler([]))
    store = ChatStore(history_turns=4)
    try:
        reply = await handle_select(client, store, 1, "  ")
    finally:
        await client.aclose()
    assert reply.startswith("Usage:")


# -- /current and /reset -------------------------------------------

def test_handle_current_unset():
    store = ChatStore(history_turns=4)
    assert "No model" in handle_current(store, 1)


def test_handle_current_set():
    store = ChatStore(history_turns=4)
    store.set_model(1, "gpt2")
    assert "gpt2" in handle_current(store, 1)


def test_handle_reset():
    store = ChatStore(history_turns=4)
    store.set_model(1, "gpt2")
    store.append(1, "user", "hi")
    msg = handle_reset(store, 1)
    assert "cleared" in msg.lower()
    assert store.messages(1) == []
    # model is preserved
    assert store.get(1).selected_model == "gpt2"


# -- plain text completion -----------------------------------------

async def test_handle_text_no_model(make_client):
    client = make_client(_models_handler([]))
    store = ChatStore(history_turns=4)
    try:
        reply = await handle_text(client, store, 1, "hi", max_tokens=8)
    finally:
        await client.aclose()
    assert "No model selected" in reply
    assert store.messages(1) == []  # nothing appended


async def test_handle_text_chat(make_client):
    handler, captured = _chat_handler("hello back")
    client = make_client(handler)
    store = ChatStore(history_turns=4)
    store.set_model(1, "gpt2")
    try:
        reply = await handle_text(client, store, 1, "hi", max_tokens=16)
    finally:
        await client.aclose()
    assert reply == "hello back"
    assert captured["body"]["model"] == "gpt2"
    assert captured["body"]["max_tokens"] == 16
    assert captured["body"]["messages"] == [
        {"role": "user", "content": "hi"},
    ]
    history = store.messages(1)
    assert history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello back"},
    ]


async def test_handle_text_includes_history(make_client):
    handler, captured = _chat_handler("ok")
    client = make_client(handler)
    store = ChatStore(history_turns=8)
    store.set_model(1, "gpt2")
    store.append(1, "user", "first")
    store.append(1, "assistant", "second")
    try:
        await handle_text(client, store, 1, "third", max_tokens=8)
    finally:
        await client.aclose()
    assert captured["body"]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]


async def test_handle_text_rolls_back_user_on_error(make_client):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")
    client = make_client(handler)
    store = ChatStore(history_turns=4)
    store.set_model(1, "gpt2")
    try:
        reply = await handle_text(client, store, 1, "hi", max_tokens=8)
    finally:
        await client.aclose()
    assert "Gateway error" in reply
    # The failed user turn is rolled back so the next retry isn't duplicated.
    assert store.messages(1) == []


async def test_handle_text_empty_returns_empty(make_client):
    client = make_client(_models_handler([]))
    store = ChatStore(history_turns=4)
    store.set_model(1, "gpt2")
    try:
        reply = await handle_text(client, store, 1, "   ", max_tokens=8)
    finally:
        await client.aclose()
    assert reply == ""


# -- callback parsing ----------------------------------------------

def test_parse_select_callback():
    assert parse_select_callback("select:gpt2") == "gpt2"
    assert parse_select_callback("noop") is None
    assert parse_select_callback("select:") == ""
