# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_gpt_neo_hf_parity.py
# Thorough GPT-Neo submodule parity vs HuggingFace (mirrors deleted NNGraph
# gpt_neo_{config,mlp,attention,decoder,model,causal} matrix).

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
from parity_helpers import (
    additive_causal_mask,
    additive_local_causal_mask,
    assert_close,
    contiguous_to_nntile,
)


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)

RTOL = 1e-4
ATOL = 1e-4
ATTN_ATOL = 5e-4
BWD_ATOL = 1e-3


# ---------------------------------------------------------------------------
# Config (was gpt_neo_config.cc)
# ---------------------------------------------------------------------------


def test_gpt_neo_config_defaults_validate_and_attention_layers():
    cfg = GPTNeoConfig(num_hidden_layers=4)
    assert cfg.head_dim == cfg.hidden_size // cfg.num_attention_heads
    assert cfg.attention_layers == ["global", "local", "global", "local"]
    cfg.validate()

    bad_hidden = GPTNeoConfig(hidden_size=66, num_attention_heads=4)
    with pytest.raises(ValueError):
        bad_hidden.validate()

    bad_layers = GPTNeoConfig(
        num_hidden_layers=2,
        attention_layers=["global"],
    )
    with pytest.raises(ValueError):
        bad_layers.validate()


def test_gpt_neo_config_from_hf_allows_alternating_layers():
    hf = _hf_cfg(attention_layers=["global", "local", "global", "local"])
    local = gpt_neo_config_from_hf(hf)
    assert local.attention_layers == ["global", "local", "global", "local"]
    assert local.head_dim == 16
    local.validate()


# ---------------------------------------------------------------------------
# Tiny HF fixtures
# ---------------------------------------------------------------------------


def _attention_types(layers: list[str]) -> list[list[object]]:
    grouped: list[list[object]] = []
    start = 0
    while start < len(layers):
        kind = layers[start]
        length = 1
        while start + length < len(layers) and layers[start + length] == kind:
            length += 1
        grouped.append([[kind], length])
        start += length
    return grouped


def _hf_cfg(
    *,
    attention_layers: list[str] | None = None,
    window_size: int = 4,
) -> HfGPTNeoConfig:
    layers = attention_layers or ["global", "global"]
    cfg = HfGPTNeoConfig(
        vocab_size=128,
        hidden_size=64,
        num_layers=len(layers),
        num_heads=4,
        max_position_embeddings=32,
        intermediate_size=128,
        window_size=window_size,
        attention_layers=list(layers),
        attention_types=_attention_types(list(layers)),
        layer_norm_epsilon=1e-5,
        activation_function="gelu_new",
        attention_dropout=0.0,
        embed_dropout=0.0,
        resid_dropout=0.0,
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _make_causal(hf_cfg: HfGPTNeoConfig):
    torch.manual_seed(0)
    hf = GPTNeoForCausalLM(hf_cfg).eval().float()
    local = GPTNeoCausal(gpt_neo_config_from_hf(hf_cfg)).eval().float()
    load_hf_into_gpt_neo_causal(local, hf)
    return hf, local.to("nntile")


def _load_attn(local: GPTNeoAttention, hf_attn: HfAttention) -> None:
    inner = hf_attn.attention
    copy_linear(local.q_proj, inner.q_proj)
    copy_linear(local.k_proj, inner.k_proj)
    copy_linear(local.v_proj, inner.v_proj)
    copy_linear(local.out_proj, inner.out_proj)


def _load_block(local: GPTNeoBlock, hf_block: HfBlock) -> None:
    local.ln_1.load_state_dict(hf_block.ln_1.state_dict())
    local.ln_2.load_state_dict(hf_block.ln_2.state_dict())
    _load_attn(local.attn, hf_block.attn)
    copy_linear(local.mlp.c_fc, hf_block.mlp.c_fc)
    copy_linear(local.mlp.c_proj, hf_block.mlp.c_proj)


def _hf_attention_reference(
    hf_attn: HfAttention,
    x: torch.Tensor,
    *,
    mode: str,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mode != "nomask":
        return hf_attn(x, attention_mask=mask)[0]

    inner = hf_attn.attention
    orig_bias = inner.bias
    inner.bias = torch.ones_like(orig_bias)
    try:
        return hf_attn(x, attention_mask=mask)[0]
    finally:
        inner.bias = orig_bias


# ---------------------------------------------------------------------------
# MLP (was gpt_neo_mlp.cc)
# ---------------------------------------------------------------------------


def test_gpt_neo_mlp_forward_backward_matches_hf():
    torch.manual_seed(1)
    hf_cfg = _hf_cfg()
    hf_mlp = HfMLP(hf_cfg.intermediate_size, hf_cfg).eval().float()
    local = GPTNeoMLP(gpt_neo_config_from_hf(hf_cfg)).eval().float()
    copy_linear(local.c_fc, hf_mlp.c_fc)
    copy_linear(local.c_proj, hf_mlp.c_proj)
    local_n = local.to("nntile")

    x = torch.randn(2, 8, hf_cfg.hidden_size, requires_grad=True)
    y_ref = hf_mlp(x)
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)

    x_n = contiguous_to_nntile(x.detach()).requires_grad_(True)
    y = local_n(x_n)
    assert_close(y, y_ref.detach(), rtol=RTOL, atol=ATOL)
    gx, gw = torch.autograd.grad(
        y,
        (x_n, local_n.c_fc.weight),
        contiguous_to_nntile(grad),
    )
    assert_close(gx, x.grad, rtol=1e-3, atol=BWD_ATOL)
    assert_close(gw, hf_mlp.c_fc.weight.grad, rtol=1e-3, atol=BWD_ATOL)


# ---------------------------------------------------------------------------
# Attention matrix: nomask, global causal, local causal window
# (was gpt_neo_attention.cc)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["nomask", "causal", "local"])
def test_gpt_neo_attention_forward_backward_matrix(mode):
    torch.manual_seed(10 + ["nomask", "causal", "local"].index(mode))
    layers = ["global", "local"] if mode == "local" else ["global", "global"]
    layer_id = 1 if mode == "local" else 0
    hf_cfg = _hf_cfg(attention_layers=layers, window_size=4)
    hf_attn = HfAttention(hf_cfg, layer_id=layer_id).eval().float()
    local_cfg = gpt_neo_config_from_hf(hf_cfg)
    local = GPTNeoAttention(local_cfg, local=(mode == "local")).eval().float()
    _load_attn(local, hf_attn)
    local_n = local.to("nntile")

    b, s, h = 2, 8, hf_cfg.hidden_size
    x = torch.randn(b, s, h, requires_grad=True)
    if mode == "nomask":
        hf_mask = torch.zeros(b, 1, s, s)
        is_causal = False
    elif mode == "causal":
        hf_mask = additive_causal_mask(b, s)
        is_causal = True
    else:
        hf_mask = additive_local_causal_mask(b, s, hf_cfg.window_size)
        is_causal = None  # local attention builds its own window mask

    y_ref = _hf_attention_reference(hf_attn, x, mode=mode, mask=hf_mask)
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)

    x_n = contiguous_to_nntile(x.detach()).requires_grad_(True)
    y = local_n(x_n, attn_mask=None, is_causal=is_causal)
    assert_close(y, y_ref.detach(), rtol=RTOL, atol=ATTN_ATOL)
    (gx,) = torch.autograd.grad(y, x_n, contiguous_to_nntile(grad))
    assert_close(gx, x.grad, rtol=1e-3, atol=BWD_ATOL)


def test_gpt_neo_block_forward_backward_matches_hf():
    torch.manual_seed(20)
    hf_cfg = _hf_cfg()
    hf_block = HfBlock(hf_cfg, layer_id=0).eval().float()
    local = GPTNeoBlock(gpt_neo_config_from_hf(hf_cfg), 0).eval().float()
    _load_block(local, hf_block)
    local_n = local.to("nntile")

    b, s, h = 2, 8, hf_cfg.hidden_size
    x = torch.randn(b, s, h, requires_grad=True)
    mask = additive_causal_mask(b, s)
    y_ref = hf_block(x, attention_mask=mask)[0]
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)

    x_n = contiguous_to_nntile(x.detach()).requires_grad_(True)
    y = local_n(x_n, attn_mask=None, is_causal=True)
    assert_close(y, y_ref.detach(), rtol=RTOL, atol=ATTN_ATOL)
    (gx,) = torch.autograd.grad(y, x_n, contiguous_to_nntile(grad))
    assert_close(gx, x.grad, rtol=1e-3, atol=BWD_ATOL)


# ---------------------------------------------------------------------------
# Model + Causal LM (was gpt_neo_model.cc / gpt_neo_causal.cc)
# ---------------------------------------------------------------------------


def test_gpt_neo_model_hidden_forward_matches_hf():
    hf_cfg = _hf_cfg(attention_layers=["global", "global"])
    hf, local = _make_causal(hf_cfg)
    ids = torch.randint(0, hf_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        ref = hf.transformer(ids).last_hidden_state
        out = local.transformer(contiguous_to_nntile(ids))
    assert_close(out, ref, rtol=RTOL, atol=ATTN_ATOL)


def test_gpt_neo_model_alternating_local_global_matches_hf():
    """Default HF pattern: even global, odd local sliding window."""
    hf_cfg = _hf_cfg(
        attention_layers=["global", "local", "global", "local"],
        window_size=4,
    )
    hf, local = _make_causal(hf_cfg)
    assert local.config.is_local_attention_layer(1)
    assert not local.config.is_local_attention_layer(0)
    ids = torch.randint(0, hf_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        ref = hf(ids).logits
        out = local(contiguous_to_nntile(ids))
    assert_close(out, ref, rtol=RTOL, atol=ATTN_ATOL)


def test_gpt_neo_causal_forward_matches_hf():
    hf_cfg = _hf_cfg(attention_layers=["global", "global"])
    hf, local = _make_causal(hf_cfg)
    ids = torch.randint(0, hf_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        ref = hf(ids).logits
        out = local(contiguous_to_nntile(ids))
    assert_close(out, ref, rtol=RTOL, atol=ATTN_ATOL)


def test_gpt_neo_causal_backward_matches_hf():
    hf_cfg = _hf_cfg(attention_layers=["global", "global"])
    hf, local = _make_causal(hf_cfg)
    for p in hf.parameters():
        p.requires_grad_(True)
    for p in local.parameters():
        p.requires_grad_(True)

    ids = torch.randint(0, hf_cfg.vocab_size, (2, 8))
    grad = torch.randn(2, 8, hf_cfg.vocab_size)
    hf(ids).logits.backward(grad)
    logits = local(contiguous_to_nntile(ids))
    (gw,) = torch.autograd.grad(
        logits,
        local.transformer.h[0].attn.q_proj.weight,
        contiguous_to_nntile(grad),
    )
    assert_close(
        gw,
        hf.transformer.h[0].attn.attention.q_proj.weight.grad,
        rtol=1e-3,
        atol=BWD_ATOL,
    )
