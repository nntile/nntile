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
    {
        # Static seq2seq decode path (no KV cache): re-runs encoder +
        # decoder per generated token.
        # Constraints (see nntile/model/t5_model.py docstring):
        #   * Must be a gated T5 FF variant (T5 v1.1 / Flan-T5).
        #   * Encoder and decoder share one seq_len (cross-attn requires
        #     enc_seq_len == dec_seq_len in nntile today).
        #   * No encoder padding mask -> heavy padding degrades output.
        #     Keep max_seq_len close to the actual prompt length.
        "id": "flan-t5-small",
        "family": "t5",
        "hf_name": "google/flan-t5-small",
        "dtype": "fp32",
        "max_seq_len": 16,
        "batch_size": 1,
    },
]

# Encoder-only models -> /v1/embeddings instead of /v1/completions.
LIVE_EMBED_MODELS = [
    {
        "id": "bert-base-uncased",
        "family": "bert",
        "hf_name": "bert-base-uncased",
        "task": "embeddings",
        "dtype": "fp32",
        "max_seq_len": 16,
        "batch_size": 1,
    },
]

# Encoder-only with masked-LM head -> /v1/fill_mask.
# `mask_token` and `expected_top` are test-only metadata, stripped
# before posting to /admin/models.
LIVE_FILL_MASK_MODELS = [
    {
        "id": "bert-fill",
        "family": "bert",
        "hf_name": "bert-base-uncased",
        "task": "fill_mask",
        "dtype": "fp32",
        "max_seq_len": 16,
        "batch_size": 1,
        "_test_mask_token": "[MASK]",
        "_test_expected_top": "fashion",
    },
    {
        # The huggingface canonical for roberta-base fill-mask is
        #   "Hello I'm a <mask> model." -> top 'male' (p=~0.33).
        "id": "roberta-fill",
        "family": "roberta",
        "hf_name": "roberta-base",
        "task": "fill_mask",
        "dtype": "fp32",
        "max_seq_len": 16,
        "batch_size": 1,
        "_test_mask_token": "<mask>",
        "_test_expected_top": "male",
    },
]


def _strip_test_meta(spec: dict) -> dict:
    return {k: v for k, v in spec.items() if not k.startswith("_test_")}


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
        all_specs = (
            LIVE_MODELS + LIVE_EMBED_MODELS + LIVE_FILL_MASK_MODELS)
        for spec in all_specs:
            payload = {**_strip_test_meta(spec), "cache_dir": cache_dir}
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


def test_v1_models_lists_all(live_setup):
    client, api_key = live_setup
    r = client.get(
        "/v1/models", headers={"Authorization": f"Bearer {api_key}"})
    assert r.status_code == 200, r.text
    ids = sorted(m["id"] for m in r.json()["data"])
    expected = sorted(
        s["id"] for s in (
            LIVE_MODELS + LIVE_EMBED_MODELS + LIVE_FILL_MASK_MODELS))
    assert ids == expected


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


@pytest.mark.parametrize(
    "model_id", [s["id"] for s in LIVE_EMBED_MODELS])
def test_v1_embeddings_per_family(live_setup, model_id):
    client, api_key = live_setup
    r = client.post(
        "/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model_id,
            "input": "The quick brown fox",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == model_id
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    vec = body["data"][0]["embedding"]
    assert isinstance(vec, list) and len(vec) > 0
    # Should be a finite, non-zero vector.
    import math
    assert all(math.isfinite(x) for x in vec)
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm > 0.0, f"zero embedding for {model_id}"
    assert body["usage"]["prompt_tokens"] > 0


def test_v1_completions_on_embedding_model_400(live_setup):
    """Encoder-only families should reject /v1/completions cleanly."""
    client, api_key = live_setup
    embed_id = LIVE_EMBED_MODELS[0]["id"]
    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": embed_id, "prompt": "x", "max_tokens": 4},
    )
    assert r.status_code == 400, r.text


def test_v1_embeddings_on_completion_model_400(live_setup):
    """Causal-LM families should reject /v1/embeddings cleanly with 400."""
    client, api_key = live_setup
    gen_id = LIVE_MODELS[0]["id"]
    r = client.post(
        "/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": gen_id, "input": "x"},
    )
    assert r.status_code == 400, r.text


@pytest.mark.parametrize(
    "spec",
    LIVE_FILL_MASK_MODELS,
    ids=[s["id"] for s in LIVE_FILL_MASK_MODELS],
)
def test_v1_fill_mask_hf_canonical(live_setup, spec):
    """The huggingface canonical demos for the encoder-LM checkpoints:

      bert-base-uncased: "Hello I'm a [MASK] model." -> top 'fashion'
      roberta-base:      "Hello I'm a <mask> model." -> top 'male'

    With per-request encoder padding mask (and for roberta also the
    HF position-id offset convention) we reproduce these tops."""
    client, api_key = live_setup
    mask_token = spec["_test_mask_token"]
    expected_top = spec["_test_expected_top"]
    prompt = f"Hello I'm a {mask_token} model."
    r = client.post(
        "/v1/fill_mask",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": spec["id"], "input": prompt, "top_k": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == spec["id"]
    assert len(body["data"]) == 1
    cands = body["data"][0]
    assert len(cands) == 5
    tokens = [c["token_str"] for c in cands]
    assert cands[0]["token_str"] == expected_top, (
        f"{spec['id']}: expected top {expected_top!r}, got {tokens}")
    assert 0.05 < cands[0]["score"] < 0.5
    assert expected_top in cands[0]["sequence"]


def test_v1_fill_mask_no_mask_returns_400(live_setup):
    client, api_key = live_setup
    model_id = LIVE_FILL_MASK_MODELS[0]["id"]
    r = client.post(
        "/v1/fill_mask",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model_id, "input": "no mask here", "top_k": 3},
    )
    assert r.status_code == 400, r.text


def test_v1_fill_mask_on_completion_model_400(live_setup):
    """Causal-LM families should reject /v1/fill_mask cleanly with 400."""
    client, api_key = live_setup
    gen_id = LIVE_MODELS[0]["id"]
    r = client.post(
        "/v1/fill_mask",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": gen_id, "input": "Hello [MASK]", "top_k": 3},
    )
    assert r.status_code == 400, r.text
