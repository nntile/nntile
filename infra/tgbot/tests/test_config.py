from __future__ import annotations

import pytest

from nntile_tgbot.config import BotConfig


def test_validate_requires_bot_token(monkeypatch):
    monkeypatch.setenv("NNTILE_TGBOT_TOKEN", "")
    monkeypatch.setenv("NNTILE_TGBOT_GATEWAY_URL", "http://x")
    monkeypatch.setenv("NNTILE_TGBOT_API_KEY", "k")
    cfg = BotConfig()
    with pytest.raises(RuntimeError, match="NNTILE_TGBOT_TOKEN"):
        cfg.validate()


def test_validate_requires_api_key(monkeypatch):
    monkeypatch.setenv("NNTILE_TGBOT_TOKEN", "t")
    monkeypatch.setenv("NNTILE_TGBOT_GATEWAY_URL", "http://x")
    monkeypatch.setenv("NNTILE_TGBOT_API_KEY", "")
    cfg = BotConfig()
    with pytest.raises(RuntimeError, match="NNTILE_TGBOT_API_KEY"):
        cfg.validate()


def test_allowed_users_parses_csv(monkeypatch):
    monkeypatch.setenv("NNTILE_TGBOT_TOKEN", "t")
    monkeypatch.setenv("NNTILE_TGBOT_GATEWAY_URL", "http://x")
    monkeypatch.setenv("NNTILE_TGBOT_API_KEY", "k")
    monkeypatch.setenv("NNTILE_TGBOT_ALLOWED_USERS", "1, 2 , 3")
    cfg = BotConfig()
    cfg.validate()
    assert cfg.allowed_user_ids == {1, 2, 3}


def test_allowed_users_rejects_non_int(monkeypatch):
    monkeypatch.setenv("NNTILE_TGBOT_TOKEN", "t")
    monkeypatch.setenv("NNTILE_TGBOT_GATEWAY_URL", "http://x")
    monkeypatch.setenv("NNTILE_TGBOT_API_KEY", "k")
    monkeypatch.setenv("NNTILE_TGBOT_ALLOWED_USERS", "1,abc")
    with pytest.raises(ValueError, match="non-int"):
        BotConfig()


def test_defaults(monkeypatch):
    for key in [
        "NNTILE_TGBOT_TOKEN",
        "NNTILE_TGBOT_GATEWAY_URL",
        "NNTILE_TGBOT_API_KEY",
        "NNTILE_TGBOT_ALLOWED_USERS",
        "NNTILE_TGBOT_HISTORY_TURNS",
        "NNTILE_TGBOT_MAX_TOKENS",
        "NNTILE_TGBOT_REQUEST_TIMEOUT",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NNTILE_TGBOT_TOKEN", "t")
    monkeypatch.setenv("NNTILE_TGBOT_API_KEY", "k")
    cfg = BotConfig()
    cfg.validate()
    assert cfg.gateway_base_url == "http://127.0.0.1:8000"
    assert cfg.history_turns == 8
    assert cfg.max_tokens == 128
    assert cfg.allowed_user_ids == set()
