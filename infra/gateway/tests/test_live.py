"""
Live integration test: drive the gateway with the real NNTileModelLoader
across three small models from three different families, all materialized
under a single CUDA-backed nntile.Context.

Requires:
  - A working nntile install for the active interpreter (StarPU + the
    nntile_core extension built for matching Python), with CUDA.
  - transformers + the chosen HF models available locally or via download.

Opt in with:
    pytest infra/gateway/tests/test_live.py --live

The test downloads the model weights on first run. Set
NNTILE_GATEWAY_LIVE_CACHE to control the HF cache dir
(default: cache_hf next to the working directory).
"""

from __future__ import annotations

import os
from typing import Generator

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.live


LIVE_MODELS = [
    {
        "id": "gpt2-small",
        "family": "gpt2",
        "hf_name": "gpt2",
        "dtype": "fp32",
        "max_seq_len": 64,
        "batch_size": 1,
    },
    {
        "id": "pythia-70m",
        "family": "gpt_neox",
        "hf_name": "EleutherAI/pythia-70m",
        "dtype": "fp32",
        "max_seq_len": 64,
        "batch_size": 1,
    },
    {
        "id": "tinyllama",
        "family": "llama",
        "hf_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dtype": "fp32",
        "max_seq_len": 64,
        "batch_size": 1,
    },
]


@pytest.fixture(scope="session")
def nntile_context() -> Generator[object, None, None]:
    nntile = pytest.importorskip("nntile")
    ctx = nntile.Context(ncpu=1, ncuda=1, ooc=0, logger=0, verbose=0)
    ctx.restrict_cuda()
    try:
        yield ctx
    finally:
        nntile.starpu.wait_for_all()
        ctx.shutdown()


@pytest.fixture(scope="session")
def live_setup(nntile_context):
    pytest.importorskip("transformers")
    from nntile_gateway.config import GatewayConfig
    from nntile_gateway.model_loader import NNTileModelLoader
    from nntile_gateway.server import build_app
    from nntile_gateway.storage.memory import InMemoryStorage
    from tests.conftest import ADMIN_TOKEN, admin_headers

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
    app = build_app(
        cfg, storage=InMemoryStorage(), loader=NNTileModelLoader())
    cache_dir = os.environ.get("NNTILE_GATEWAY_LIVE_CACHE", "cache_hf")

    client = TestClient(app)
    client.__enter__()
    try:
        for spec in LIVE_MODELS:
            payload = {**spec, "cache_dir": cache_dir}
            r = client.post(
                "/admin/models", json=payload, headers=admin_headers())
            assert r.status_code == 200, (
                f"failed to register {spec['id']}: {r.status_code} {r.text}"
            )
            assert r.json()["status"] == "ready"

        r = client.post(
            "/admin/keys", json={"name": "live"}, headers=admin_headers())
        api_key = r.json()["key"]
        yield client, api_key
    finally:
        client.__exit__(None, None, None)


def test_v1_models_lists_all_three(live_setup):
    client, api_key = live_setup
    r = client.get(
        "/v1/models", headers={"Authorization": f"Bearer {api_key}"})
    assert r.status_code == 200, r.text
    ids = sorted(m["id"] for m in r.json()["data"])
    assert ids == sorted(s["id"] for s in LIVE_MODELS)


@pytest.mark.parametrize("model_id", [s["id"] for s in LIVE_MODELS])
def test_v1_completions_per_family(live_setup, model_id):
    client, api_key = live_setup
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model_id,
            "prompt": "The quick brown fox",
            "max_tokens": 8,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == model_id
    choice = body["choices"][0]
    assert isinstance(choice["text"], str)
    assert choice["text"] != "", f"empty completion for {model_id}"
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["completion_tokens"] > 0
