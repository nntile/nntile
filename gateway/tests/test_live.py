"""
Live integration test: drive the gateway with the real NNTileModelLoader.

Requires:
  - A working nntile install for the active interpreter (StarPU + the
    nntile_core extension built for matching Python).
  - transformers + the chosen HF model available locally or via download.

Opt in with:
    pytest gateway/tests/test_live.py --live

The test downloads gpt2 (~500MB) the first time. Set NNTILE_GATEWAY_LIVE_CACHE
to control the HF cache dir.
"""

from __future__ import annotations

import os
from typing import Generator

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.live


@pytest.fixture(scope="session")
def nntile_context() -> Generator[object, None, None]:
    nntile = pytest.importorskip("nntile")
    ctx = nntile.Context(ncpu=1, ncuda=0, ooc=0, logger=0, verbose=0)
    ctx.restrict_cpu()
    try:
        yield ctx
    finally:
        nntile.starpu.wait_for_all()
        ctx.shutdown()


@pytest.fixture
def live_app(nntile_context):
    pytest.importorskip("transformers")
    from nntile_gateway.config import GatewayConfig
    from nntile_gateway.model_loader import NNTileModelLoader
    from nntile_gateway.server import build_app
    from nntile_gateway.storage.memory import InMemoryStorage
    from tests.conftest import ADMIN_TOKEN

    cfg = GatewayConfig(
        admin_token=ADMIN_TOKEN,
        host="127.0.0.1",
        port=0,
        storage="memory",
        sqlite_path="",
        auth_cache_ttl=60,
        auth_cache_size=8,
        ncpu=-1,
        ncuda=-1,
    )
    storage = InMemoryStorage()
    loader = NNTileModelLoader()
    return build_app(cfg, storage=storage, loader=loader)


def test_live_gpt2_end_to_end(live_app):
    from tests.conftest import admin_headers

    cache_dir = os.environ.get(
        "NNTILE_GATEWAY_LIVE_CACHE", "cache_hf")
    spec = {
        "id": "gpt2-small",
        "family": "gpt2",
        "hf_name": "gpt2",
        "dtype": "fp32",
        "max_seq_len": 64,
        "batch_size": 1,
        "cache_dir": cache_dir,
    }

    with TestClient(live_app) as c:
        r = c.post("/admin/models", json=spec, headers=admin_headers())
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "ready"

        r = c.post(
            "/admin/keys", json={"name": "live"}, headers=admin_headers())
        api_key = r.json()["key"]
        user = {"Authorization": f"Bearer {api_key}"}

        r = c.get("/v1/models", headers=user)
        assert r.status_code == 200
        assert [m["id"] for m in r.json()["data"]] == ["gpt2-small"]

        r = c.post(
            "/v1/completions",
            headers=user,
            json={
                "model": "gpt2-small",
                "prompt": "The quick brown fox",
                "max_tokens": 8,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == "gpt2-small"
        choice = body["choices"][0]
        assert isinstance(choice["text"], str) and choice["text"] != ""
        assert body["usage"]["prompt_tokens"] > 0
        assert body["usage"]["completion_tokens"] > 0
