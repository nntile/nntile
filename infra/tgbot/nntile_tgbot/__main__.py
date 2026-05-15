"""Entrypoint: starts long polling against the Telegram Bot API."""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from nntile_tgbot.client import GatewayClient
from nntile_tgbot.config import BotConfig
from nntile_tgbot.handlers import build_router
from nntile_tgbot.httpx_session import HttpxSession
from nntile_tgbot.state import ChatStore

log = logging.getLogger(__name__)


async def _run() -> None:
    cfg = BotConfig()
    cfg.validate()

    client = GatewayClient(
        base_url=cfg.gateway_base_url,
        api_key=cfg.gateway_api_key,
        timeout_s=cfg.request_timeout_s,
    )
    store = ChatStore(history_turns=cfg.history_turns)

    # We always use HttpxSession; httpx natively handles `https://`
    # proxies (TLS to the proxy itself), which is what the corporate
    # proxy used here requires. aiogram's stock AiohttpSession routes
    # through aiohttp_socks which only supports plain HTTP/SOCKS to
    # the proxy.
    if cfg.telegram_proxy:
        log.info("using proxy for Telegram API: %s",
                 _redact_proxy(cfg.telegram_proxy))
    session = HttpxSession(proxy=cfg.telegram_proxy)
    bot = Bot(token=cfg.bot_token, session=session)
    dp = Dispatcher()
    dp.include_router(build_router(client, store, cfg))

    try:
        await dp.start_polling(bot)
    finally:
        await client.aclose()
        await bot.session.close()


def _redact_proxy(url: str) -> str:
    """Hide creds in the proxy URL when logging."""
    import re
    return re.sub(r"://[^@]+@", "://***@", url)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_run())


if __name__ == "__main__":
    main()
