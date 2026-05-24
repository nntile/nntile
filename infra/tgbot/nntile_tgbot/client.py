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
    """One entry from `GET /v1/models`.

    `family`, `task`, and `max_seq_len` are gateway-side extensions
    (not part of the OpenAI shape). The bot uses `max_seq_len` to
    cap outgoing `max_tokens` automatically on `/select`."""

    id: str
    family: str | None
    status: str | None
    task: str | None = None
    max_seq_len: int | None = None


@dataclass(frozen=True)
class FillMaskCandidate:
    """One predicted token for a single [MASK] position.

    Mirrors the gateway's response shape; the handler renders these
    as numbered lines with token, score, and the filled-in sequence."""

    token: int
    token_str: str
    score: float
    sequence: str


class GatewayClient:
    """Async wrapper around the gateway HTTP API.

    Owns its httpx.AsyncClient; the caller must close it via aclose()."""

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
        """Release the underlying httpx connection pool."""
        await self._client.aclose()

    async def list_models(self) -> list[ModelInfo]:
        """`GET /v1/models` -> [ModelInfo]. Raises `GatewayError`."""
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
                task=entry.get("task"),
                max_seq_len=entry.get("max_seq_len"),
            ))
        return out

    async def fill_mask(
        self,
        model: str,
        text: str,
        top_k: int = 5,
    ) -> list[list[FillMaskCandidate]]:
        """Returns one inner list of candidates per [MASK] position."""
        r = await self._client.post(
            f"{self._base_url}/v1/fill_mask",
            headers=self._headers,
            json={"model": model, "input": text, "top_k": top_k},
        )
        self._raise_for_status(r)
        payload = r.json()
        return [
            [FillMaskCandidate(
                token=c["token"],
                token_str=c["token_str"],
                score=c["score"],
                sequence=c["sequence"],
             ) for c in per_mask]
            for per_mask in payload.get("data", [])
        ]

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """`POST /v1/chat/completions` -> assistant message text."""
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
