"""Test helpers: build a GatewayClient backed by httpx.MockTransport."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest
from nntile_tgbot.client import GatewayClient


@pytest.fixture
def make_client() -> Callable[[Callable[[httpx.Request], httpx.Response]],
                              GatewayClient]:
    """Factory: handler -> GatewayClient routed to that handler."""

    def _factory(
        handler: Callable[[httpx.Request], httpx.Response],
    ) -> GatewayClient:
        transport = httpx.MockTransport(handler)
        return GatewayClient(
            base_url="http://test",
            api_key="nnt_test",
            client=httpx.AsyncClient(transport=transport, trust_env=False),
        )

    return _factory
