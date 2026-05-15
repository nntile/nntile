"""
Minimal aiogram session backed by httpx.

We need this because aiogram's stock AiohttpSession routes proxies
through aiohttp_socks, which only speaks `http://` / `socks*://`
schemes -- it can't TLS-wrap the connection to the proxy. The corporate
HTTPS proxy used in this environment (https://...:4443) requires TLS to
the proxy itself, which httpx supports natively via `proxy=https://...`.

This implementation covers the methods our bot exercises: simple
non-file POSTs (sendMessage, getMe, getUpdates, answerCallbackQuery,
etc.). File uploads aren't supported -- the bot doesn't send files.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Optional, cast

import httpx
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramNetworkError
from aiogram.methods import TelegramMethod
from aiogram.methods.base import TelegramType


class HttpxSession(BaseSession):
    def __init__(self, *args, proxy: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._proxy = proxy
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            kwargs: dict[str, Any] = {"timeout": self.timeout}
            if self._proxy:
                kwargs["proxy"] = self._proxy
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_data(
        self, bot: Bot, method: TelegramMethod[TelegramType],
    ) -> dict[str, Any]:
        files: dict[str, Any] = {}
        out: dict[str, Any] = {}
        for key, raw in method.model_dump(warnings=False).items():
            value = self.prepare_value(raw, bot=bot, files=files)
            if value in (None, "", [], {}):
                continue
            out[key] = value
        if files:
            raise NotImplementedError(
                "HttpxSession does not support file uploads; the bot "
                "does not currently send files. If you need this, "
                "switch to AiohttpSession (and find a non-TLS proxy) "
                "or extend this session with multipart support.")
        return out

    async def make_request(
        self,
        bot: Bot,
        method: TelegramMethod[TelegramType],
        timeout: Optional[int] = None,
    ) -> TelegramType:
        client = await self._get_client()
        url = self.api.api_url(
            token=bot.token, method=method.__api_method__)
        data = self._build_data(bot=bot, method=method)
        try:
            resp = await client.post(
                url,
                data=data,
                timeout=(
                    self.timeout if timeout is None else timeout),
            )
            raw_result = resp.text
        except httpx.TimeoutException as e:
            raise TelegramNetworkError(
                method=method, message="Request timeout error") from e
        except httpx.HTTPError as e:
            raise TelegramNetworkError(
                method=method,
                message=f"{type(e).__name__}: {e}",
            ) from e
        response = self.check_response(
            bot=bot, method=method,
            status_code=resp.status_code, content=raw_result,
        )
        return cast(TelegramType, response.result)

    async def stream_content(
        self,
        url: str,
        headers: Optional[dict[str, Any]] = None,
        timeout: int = 30,
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        client = await self._get_client()
        async with client.stream(
            "GET", url, headers=headers or {}, timeout=timeout,
        ) as r:
            if raise_for_status:
                r.raise_for_status()
            async for chunk in r.aiter_bytes(chunk_size=chunk_size):
                yield chunk
