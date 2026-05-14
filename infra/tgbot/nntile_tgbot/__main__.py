"""Entrypoint: starts long polling against the Telegram Bot API."""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher

from nntile_tgbot.client import GatewayClient
from nntile_tgbot.config import BotConfig
from nntile_tgbot.handlers import build_router
from nntile_tgbot.state import ChatStore


async def _run() -> None:
    cfg = BotConfig()
    cfg.validate()

    client = GatewayClient(
        base_url=cfg.gateway_base_url,
        api_key=cfg.gateway_api_key,
        timeout_s=cfg.request_timeout_s,
    )
    store = ChatStore(history_turns=cfg.history_turns)

    bot = Bot(token=cfg.bot_token)
    dp = Dispatcher()
    dp.include_router(build_router(client, store, cfg))

    try:
        await dp.start_polling(bot)
    finally:
        await client.aclose()
        await bot.session.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_run())


if __name__ == "__main__":
    main()
