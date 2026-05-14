"""Async HTTP client around the nntile-gateway OpenAI-compatible API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class GatewayError(RuntimeError):
    """The gateway returned a non-2xx status."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"gateway {status_code}: {body[:200]}")
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class ModelInfo:
    id: str
    family: str | None
    status: str | None


class GatewayClient:
    """Thin async wrapper. Owns its httpx.AsyncClient; close with `aclose()`."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        # trust_env=False prevents corporate HTTP_PROXY from intercepting
        # localhost traffic; matches the gateway's live test client.
        self._client = client or httpx.AsyncClient(
            timeout=timeout_s, trust_env=False)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[ModelInfo]:
        r = await self._client.get(
            f"{self._base_url}/v1/models", headers=self._headers)
        self._raise_for_status(r)
        data = r.json().get("data", [])
        out: list[ModelInfo] = []
        for entry in data:
            out.append(ModelInfo(
                id=entry["id"],
                family=entry.get("family"),
                status=entry.get("status"),
            ))
        return out

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        r = await self._client.post(
            f"{self._base_url}/v1/chat/completions",
            headers=self._headers,
            json=body,
        )
        self._raise_for_status(r)
        payload = r.json()
        return payload["choices"][0]["message"]["content"]

    @staticmethod
    def _raise_for_status(r: httpx.Response) -> None:
        if r.status_code // 100 != 2:
            raise GatewayError(r.status_code, r.text)
