"""
aiogram v3 router that delegates to the pure functions in nntile_tgbot.core.

The router is built by `build_router(client, store, config)` so we can wire
the shared dependencies via closure instead of aiogram's DI machinery — it
keeps the handlers easy to read and the testable logic lives in core.py.
"""

from __future__ import annotations

import logging

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from nntile_tgbot.client import GatewayClient
from nntile_tgbot.config import BotConfig
from nntile_tgbot.core import (
    WELCOME,
    handle_current,
    handle_fill_mask,
    handle_models,
    handle_reset,
    handle_select,
    handle_text,
    is_authorized,
    parse_select_callback,
)
from nntile_tgbot.state import ChatStore

log = logging.getLogger(__name__)


def _keyboard(reply) -> InlineKeyboardMarkup | None:
    if not reply.buttons:
        return None
    rows = [[
        InlineKeyboardButton(text=b.label, callback_data=b.callback_data)
    ] for b in reply.buttons]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_router(
    client: GatewayClient,
    store: ChatStore,
    config: BotConfig,
) -> Router:
    router = Router()
    allowed = set(config.allowed_user_ids)
    max_tokens = config.max_tokens

    def _gate(user_id: int | None) -> bool:
        return is_authorized(user_id, allowed)

    @router.message(CommandStart())
    @router.message(Command("help"))
    async def cmd_start_help(message: Message) -> None:
        if not _gate(message.from_user.id if message.from_user else None):
            return
        await message.answer(WELCOME)

    @router.message(Command("models"))
    async def cmd_models(message: Message) -> None:
        if not _gate(message.from_user.id if message.from_user else None):
            return
        reply = await handle_models(client)
        await message.answer(reply.text, reply_markup=_keyboard(reply))

    @router.message(Command("select"))
    async def cmd_select(message: Message) -> None:
        if not _gate(message.from_user.id if message.from_user else None):
            return
        text = (message.text or "").split(maxsplit=1)
        arg = text[1] if len(text) == 2 else ""
        reply = await handle_select(client, store, message.chat.id, arg)
        await message.answer(reply)

    @router.message(Command("current"))
    async def cmd_current(message: Message) -> None:
        if not _gate(message.from_user.id if message.from_user else None):
            return
        await message.answer(handle_current(store, message.chat.id))

    @router.message(Command("reset"))
    async def cmd_reset(message: Message) -> None:
        if not _gate(message.from_user.id if message.from_user else None):
            return
        await message.answer(handle_reset(store, message.chat.id))

    @router.message(Command("fill"))
    async def cmd_fill(message: Message) -> None:
        if not _gate(message.from_user.id if message.from_user else None):
            return
        text = (message.text or "").split(maxsplit=1)
        arg = text[1] if len(text) == 2 else ""
        reply = await handle_fill_mask(
            client, store, message.chat.id, arg, top_k=5)
        await message.answer(reply)

    @router.callback_query()
    async def on_callback(cq: CallbackQuery) -> None:
        if not _gate(cq.from_user.id if cq.from_user else None):
            await cq.answer()
            return
        data = cq.data or ""
        model_id = parse_select_callback(data)
        if model_id is None:
            await cq.answer()
            return
        reply = await handle_select(
            client, store, cq.message.chat.id, model_id)
        await cq.answer(reply, show_alert=False)
        # Also send a visible confirmation in chat.
        await cq.message.answer(reply)

    @router.message()
    async def on_text(message: Message) -> None:
        if not _gate(message.from_user.id if message.from_user else None):
            return
        if not message.text:
            return
        reply = await handle_text(
            client,
            store,
            message.chat.id,
            message.text,
            max_tokens,
        )
        if reply:
            await message.answer(reply)

    return router
