# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_gpt_neo_hf_parity.py
# GPT-Neo layer + full-model parity vs HuggingFace GPTNeoForCausalLM.

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("transformers")

import torch
from transformers import GPTNeoConfig as HfGPTNeoConfig
from transformers import GPTNeoForCausalLM
from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoAttention as HfAttention,
    GPTNeoBlock as HfBlock,
    GPTNeoMLP as HfMLP,
)

import torch_nntile
from torch_nntile import _C
from torch_nntile.models.gpt_neo import (
    GPTNeoAttention,
    GPTNeoBlock,
    GPTNeoCausal,
    GPTNeoConfig,
    GPTNeoMLP,
)
from torch_nntile.models.gpt_neo_hf_loader import (
    gpt_neo_config_from_hf,
    load_hf_into_gpt_neo_causal,
)
from torch_nntile.models.hf_rope_layout import copy_linear
from parity_helpers import assert_close, contiguous_to_nntile


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)

RTOL = 1e-4
ATOL = 1e-4


@pytest.fixture
def tiny_hf_config() -> HfGPTNeoConfig:
    # All-global: local window masks are not auto-built in torch_nntile yet.
    n_layer = 2
    cfg = HfGPTNeoConfig(
        vocab_size=128,
        hidden_size=64,
        num_layers=n_layer,
        num_heads=4,
        max_position_embeddings=32,
        intermediate_size=128,
        window_size=256,
        attention_layers=["global"] * n_layer,
        attention_types=[[["global"], n_layer]],
        layer_norm_epsilon=1e-5,
        activation_function="gelu_new",
        attention_dropout=0.0,
        embed_dropout=0.0,
        resid_dropout=0.0,
        tie_word_embeddings=True,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _make_models(hf_cfg: HfGPTNeoConfig):
    torch.manual_seed(0)
    hf = GPTNeoForCausalLM(hf_cfg).eval().float()
    local_cfg = gpt_neo_config_from_hf(hf_cfg)
    minimal = GPTNeoCausal(local_cfg).eval().float()
    load_hf_into_gpt_neo_causal(minimal, hf)
    minimal = minimal.to("nntile")
    return hf, minimal


def _load_attn(local: GPTNeoAttention, hf_attn: HfAttention) -> None:
    inner = hf_attn.attention
    copy_linear(local.q_proj, inner.q_proj)
    copy_linear(local.k_proj, inner.k_proj)
    copy_linear(local.v_proj, inner.v_proj)
    copy_linear(local.out_proj, inner.out_proj)


def test_gpt_neo_mlp_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(1)
    hf_mlp = HfMLP(tiny_hf_config.intermediate_size, tiny_hf_config).eval().float()
    local = GPTNeoMLP(gpt_neo_config_from_hf(tiny_hf_config)).eval().float()
    copy_linear(local.c_fc, hf_mlp.c_fc)
    copy_linear(local.c_proj, hf_mlp.c_proj)
    local = local.to("nntile")
    x = torch.randn(2, 8, tiny_hf_config.hidden_size)
    with torch.no_grad():
        ref = hf_mlp(x)
        out = local(contiguous_to_nntile(x))
    assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_gpt_neo_mlp_backward_matches_hf(tiny_hf_config):
    torch.manual_seed(2)
    hf_mlp = HfMLP(tiny_hf_config.intermediate_size, tiny_hf_config).eval().float()
    local = GPTNeoMLP(gpt_neo_config_from_hf(tiny_hf_config)).eval().float()
    copy_linear(local.c_fc, hf_mlp.c_fc)
    copy_linear(local.c_proj, hf_mlp.c_proj)
    local = local.to("nntile")
    x = torch.randn(2, 8, tiny_hf_config.hidden_size, requires_grad=True)
    grad = torch.randn_like(x)
    y_ref = hf_mlp(x)
    y_ref.backward(grad)
    x_nnt = contiguous_to_nntile(x.detach()).requires_grad_(True)
    y = local(x_nnt)
    (gx,) = torch.autograd.grad(
        y, x_nnt, grad_outputs=contiguous_to_nntile(grad)
    )
    assert_close(gx, x.grad, rtol=RTOL, atol=ATOL)


def test_gpt_neo_attention_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(3)
    hf_attn = HfAttention(tiny_hf_config, layer_id=0).eval().float()
    local = GPTNeoAttention(
        gpt_neo_config_from_hf(tiny_hf_config), local=False
    ).eval().float()
    _load_attn(local, hf_attn)
    local = local.to("nntile")
    x = torch.randn(2, 8, tiny_hf_config.hidden_size)
    with torch.no_grad():
        ref = hf_attn(x)[0]
        out = local(contiguous_to_nntile(x), attn_mask=None)
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


def test_gpt_neo_block_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(4)
    hf_block = HfBlock(tiny_hf_config, layer_id=0).eval().float()
    local = GPTNeoBlock(gpt_neo_config_from_hf(tiny_hf_config), 0).eval().float()
    local.ln_1.load_state_dict(hf_block.ln_1.state_dict())
    local.ln_2.load_state_dict(hf_block.ln_2.state_dict())
    _load_attn(local.attn, hf_block.attn)
    copy_linear(local.mlp.c_fc, hf_block.mlp.c_fc)
    copy_linear(local.mlp.c_proj, hf_block.mlp.c_proj)
    local = local.to("nntile")
    x = torch.randn(2, 8, tiny_hf_config.hidden_size)
    with torch.no_grad():
        ref = hf_block(x)[0]
        out = local(contiguous_to_nntile(x), attn_mask=None)
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


def test_gpt_neo_causal_forward_matches_hf(tiny_hf_config):
    hf, minimal = _make_models(tiny_hf_config)
    input_ids = torch.randint(0, tiny_hf_config.vocab_size, (2, 8))
    with torch.no_grad():
        ref = hf(input_ids).logits
        out = minimal(contiguous_to_nntile(input_ids))
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


def test_gpt_neo_causal_backward_matches_hf(tiny_hf_config):
    hf, minimal = _make_models(tiny_hf_config)
    for p in hf.parameters():
        p.requires_grad_(True)
    for p in minimal.parameters():
        p.requires_grad_(True)
    input_ids = torch.randint(0, tiny_hf_config.vocab_size, (2, 8))
    grad = torch.randn(2, 8, tiny_hf_config.vocab_size)
    logits_ref = hf(input_ids).logits
    logits_ref.backward(grad)
    logits = minimal(contiguous_to_nntile(input_ids))
    (gw,) = torch.autograd.grad(
        logits,
        minimal.transformer.wte.weight,
        grad_outputs=contiguous_to_nntile(grad),
    )
    assert_close(gw, hf.transformer.wte.weight.grad, rtol=1e-3, atol=1e-3)
