# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_hf_gpt2_stock_on_nntile.py
# Stock HuggingFace GPT2LMHeadModel on device="nntile".

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("transformers")

import torch
from transformers import GPT2Config, GPT2LMHeadModel

import torch_nntile
from torch_nntile import _C
from torch_nntile.training import cross_entropy, train_full_batch_step
from conftest import nntile_cpu


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


_SKIP_FULL_MODEL_GRAPH = pytest.mark.skip(
    reason="Full GPT-2 TensorGraph execute aborts (uninitialized handles in add)",
)


@pytest.fixture(scope="module", autouse=True)
def _nntile_context_no_fallback():
    if not _C.has_libnntile():
        return
    if torch_nntile.is_cpu_fallback_enabled():
        pytest.skip(
            "context has CPU fallback enabled; rebuild with cpu_fallback=False"
        )
    if not torch_nntile.is_context_initialized():
        torch_nntile.init_context(
            ncpu=1,
            ncuda=0,
            verbose=0,
            cpu_fallback=False,
        )
    torch_nntile.restrict_cpu()
    yield


@pytest.fixture
def tiny_gpt2_config() -> GPT2Config:
    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        n_positions=32,
        vocab_size=128,
        n_inner=256,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "sdpa"
    return config


def _make_stock_models(config: GPT2Config):
    torch.manual_seed(0)
    ref = GPT2LMHeadModel(config).eval().float()
    model = GPT2LMHeadModel(config).eval().float()
    model.load_state_dict(ref.state_dict())
    model = model.to("nntile")
    return ref, model


@_SKIP_FULL_MODEL_GRAPH
def test_hf_gpt2_forward_matches_cpu(tiny_gpt2_config):
    ref, model = _make_stock_models(tiny_gpt2_config)
    input_ids = torch.randint(0, tiny_gpt2_config.vocab_size, (2, 8)).to("nntile")
    with torch.no_grad():
        ref_logits = ref(nntile_cpu(input_ids)).logits
        out = model(input_ids).logits
    torch.testing.assert_close(nntile_cpu(out), ref_logits, rtol=1e-4, atol=1e-4)


@_SKIP_FULL_MODEL_GRAPH
def test_hf_gpt2_cross_entropy_backward_matches_cpu(tiny_gpt2_config):
    ref, model = _make_stock_models(tiny_gpt2_config)
    for param in ref.parameters():
        param.requires_grad_(True)
    for param in model.parameters():
        param.requires_grad_(True)

    input_ids = torch.randint(0, tiny_gpt2_config.vocab_size, (2, 8)).to("nntile")
    labels = input_ids.clone()

    ref.zero_grad(set_to_none=True)
    ref_logits = ref(nntile_cpu(input_ids)).logits
    ref_loss = torch.nn.functional.cross_entropy(
        ref_logits.view(-1, tiny_gpt2_config.vocab_size),
        nntile_cpu(labels).view(-1),
    )
    ref_loss.backward()

    model.zero_grad(set_to_none=True)
    logits = model(input_ids).logits
    loss = cross_entropy(logits, labels, reduction="mean")
    gw_nnt, = torch.autograd.grad(loss, model.transformer.wte.weight)

    torch.testing.assert_close(nntile_cpu(loss), ref_loss, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        nntile_cpu(gw_nnt),
        ref.transformer.wte.weight.grad,
        rtol=1e-3,
        atol=1e-3,
    )


@_SKIP_FULL_MODEL_GRAPH
def test_hf_gpt2_train_full_batch_step_nntile_inputs(tiny_gpt2_config):
    _, model = _make_stock_models(tiny_gpt2_config)
    for param in model.parameters():
        param.requires_grad_(True)

    input_ids = torch.randint(0, tiny_gpt2_config.vocab_size, (2, 8)).to("nntile")
    labels = input_ids.clone()
    loss = train_full_batch_step(model, input_ids, labels, learning_rate=1e-3)
    assert loss > 0.0
    assert model.transformer.wte.weight.grad is not None
