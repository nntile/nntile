"""Lightweight smoke test for handlers.build_router.

The full handler behavior is covered by tests/test_core.py against the
plain functions; here we just ensure the aiogram Router wires up without
raising and the inline keyboard builder produces the expected structure.
"""

from __future__ import annotations

from aiogram import Router
from aiogram.types import InlineKeyboardMarkup

from nntile_tgbot.config import BotConfig
from nntile_tgbot.core import ModelButton, ModelListReply
from nntile_tgbot.handlers import _keyboard, build_router
from nntile_tgbot.state import ChatStore


def test_build_router_returns_router(make_client):
    client = make_client(
        lambda req: __import__("httpx").Response(200, json={"data": []}))
    cfg = BotConfig(
        bot_token="x",
        gateway_base_url="http://x",
        gateway_api_key="k",
        allowed_user_ids=set(),
        history_turns=4,
        max_tokens=8,
        request_timeout_s=10.0,
    )
    store = ChatStore(history_turns=4)
    router = build_router(client, store, cfg)
    assert isinstance(router, Router)


def test_keyboard_none_when_no_buttons():
    reply = ModelListReply(text="x", buttons=[])
    assert _keyboard(reply) is None


def test_keyboard_one_row_per_button():
    reply = ModelListReply(text="x", buttons=[
        ModelButton(label="gpt2", callback_data="select:gpt2"),
        ModelButton(label="tiny", callback_data="select:tiny"),
    ])
    kb = _keyboard(reply)
    assert isinstance(kb, InlineKeyboardMarkup)
    assert len(kb.inline_keyboard) == 2
    assert kb.inline_keyboard[0][0].text == "gpt2"
    assert kb.inline_keyboard[0][0].callback_data == "select:gpt2"
    assert kb.inline_keyboard[1][0].text == "tiny"
