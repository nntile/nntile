"""Env-driven configuration for the Telegram bot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name, default)
    if val == "":
        return None
    return val


def _parse_int_set(raw: str | None) -> set[int]:
    if not raw:
        return set()
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            raise ValueError(
                f"NNTILE_TGBOT_ALLOWED_USERS contains non-int entry: {part!r}"
            )
    return out


@dataclass
class BotConfig:
    bot_token: str = field(
        default_factory=lambda: _env("NNTILE_TGBOT_TOKEN", "") or "")
    gateway_base_url: str = field(
        default_factory=lambda: _env(
            "NNTILE_TGBOT_GATEWAY_URL", "http://127.0.0.1:8000") or "")
    gateway_api_key: str = field(
        default_factory=lambda: _env("NNTILE_TGBOT_API_KEY", "") or "")
    # Comma-separated list of Telegram user IDs allowed to use the bot.
    # Empty = open to anyone who can reach the bot (still gated by gateway key).
    allowed_user_ids: set[int] = field(
        default_factory=lambda: _parse_int_set(
            _env("NNTILE_TGBOT_ALLOWED_USERS")))
    # Max messages of history sent back to the gateway per turn (incl. system).
    history_turns: int = field(
        default_factory=lambda: int(_env("NNTILE_TGBOT_HISTORY_TURNS", "8")))
    # Max tokens to request from the gateway.
    max_tokens: int = field(
        default_factory=lambda: int(_env("NNTILE_TGBOT_MAX_TOKENS", "128")))
    # Per-request timeout in seconds.
    request_timeout_s: float = field(
        default_factory=lambda: float(
            _env("NNTILE_TGBOT_REQUEST_TIMEOUT", "120")))
    # HTTP(S) proxy URL for Telegram API calls. The gateway client
    # always uses trust_env=False so this proxy never intercepts
    # localhost traffic. Falls back to HTTPS_PROXY / HTTP_PROXY env.
    # We use HttpxSession (in __main__) for the bot, so an `https://`
    # proxy URL works -- TLS to the proxy is handled natively by httpx.
    telegram_proxy: str | None = field(
        default_factory=lambda: (
            _env("NNTILE_TGBOT_TELEGRAM_PROXY")
            or _env("HTTPS_PROXY")
            or _env("HTTP_PROXY")
        ))

    def validate(self) -> None:
        if not self.bot_token:
            raise RuntimeError("NNTILE_TGBOT_TOKEN is required")
        if not self.gateway_base_url:
            raise RuntimeError("NNTILE_TGBOT_GATEWAY_URL is required")
        if not self.gateway_api_key:
            raise RuntimeError("NNTILE_TGBOT_API_KEY is required")
        if self.history_turns < 1:
            raise RuntimeError("history_turns must be >= 1")
        if self.max_tokens < 1:
            raise RuntimeError("max_tokens must be >= 1")
