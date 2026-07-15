# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_llama_hf_parity.py
# Llama layer + full-model parity vs HuggingFace LlamaForCausalLM.

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("transformers")

import torch
from transformers import LlamaConfig as HfLlamaConfig
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaAttention as HfAttention,
    LlamaDecoderLayer as HfDecoder,
    LlamaMLP as HfMLP,
    LlamaRotaryEmbedding,
)

import torch_nntile
from torch_nntile import _C
from conftest import nntile_cpu
from torch_nntile.models.llama import (
    LlamaAttention,
    LlamaCausal,
    LlamaConfig,
    LlamaDecoder,
    LlamaMLP,
)
from torch_nntile.models.llama_hf_loader import (
    llama_config_from_hf,
    load_hf_into_llama_causal,
)
from torch_nntile.models.hf_rope_layout import (
    copy_linear,
    hf_to_nntile_qkv_weight,
)
from parity_helpers import assert_close, contiguous_to_nntile


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)

RTOL = 1e-4
ATOL = 1e-4


@pytest.fixture
def tiny_hf_config() -> HfLlamaConfig:
    cfg = HfLlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        hidden_act="silu",
    )
    cfg._attn_implementation = "eager"
    return cfg


@pytest.fixture
def tiny_gqa_hf_config(tiny_hf_config) -> HfLlamaConfig:
    tiny_hf_config.num_key_value_heads = 2
    return tiny_hf_config


def _make_models(hf_cfg: HfLlamaConfig):
    torch.manual_seed(0)
    hf = LlamaForCausalLM(hf_cfg).eval().float()
    local_cfg = llama_config_from_hf(hf_cfg)
    minimal = LlamaCausal(local_cfg).eval().float()
    load_hf_into_llama_causal(minimal, hf)
    minimal = minimal.to("nntile")
    return hf, minimal


def test_llama_mlp_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(1)
    hf_mlp = HfMLP(tiny_hf_config).eval().float()
    local = LlamaMLP(llama_config_from_hf(tiny_hf_config)).eval().float()
    copy_linear(local.gate_proj, hf_mlp.gate_proj)
    copy_linear(local.up_proj, hf_mlp.up_proj)
    copy_linear(local.down_proj, hf_mlp.down_proj)
    local = local.to("nntile")
    x = torch.randn(2, 8, tiny_hf_config.hidden_size)
    with torch.no_grad():
        ref = hf_mlp(x)
        out = local(contiguous_to_nntile(x))
    assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_llama_mlp_backward_matches_hf(tiny_hf_config):
    torch.manual_seed(2)
    hf_mlp = HfMLP(tiny_hf_config).eval().float()
    local = LlamaMLP(llama_config_from_hf(tiny_hf_config)).eval().float()
    copy_linear(local.gate_proj, hf_mlp.gate_proj)
    copy_linear(local.up_proj, hf_mlp.up_proj)
    copy_linear(local.down_proj, hf_mlp.down_proj)
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


def _load_attn_local(local: LlamaAttention, hf_attn, cfg: LlamaConfig) -> None:
    local.q_proj.weight.data.copy_(
        hf_to_nntile_qkv_weight(
            hf_attn.q_proj.weight.data,
            n_heads=cfg.num_attention_heads,
            head_dim=cfg.head_dim,
        )
    )
    local.k_proj.weight.data.copy_(
        hf_to_nntile_qkv_weight(
            hf_attn.k_proj.weight.data,
            n_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
        )
    )
    copy_linear(local.v_proj, hf_attn.v_proj)
    copy_linear(local.o_proj, hf_attn.o_proj)


def _hf_position_embeddings(hf_cfg: HfLlamaConfig, position_ids: torch.Tensor):
    """Return HF ``(cos, sin)`` full-dim tables for ``position_ids``."""
    rotary = LlamaRotaryEmbedding(config=hf_cfg)
    # Dummy hidden used only for dtype/device; values unused by rotary.
    hidden = torch.zeros(
        position_ids.shape[0],
        position_ids.shape[1],
        hf_cfg.hidden_size,
        dtype=torch.float32,
    )
    return rotary(hidden, position_ids)


@pytest.mark.parametrize("gqa", [False, True])
def test_llama_attention_forward_matches_hf(tiny_hf_config, gqa):
    torch.manual_seed(3)
    if gqa:
        tiny_hf_config.num_key_value_heads = 2
    hf_attn = HfAttention(tiny_hf_config, layer_idx=0).eval().float()
    local_cfg = llama_config_from_hf(tiny_hf_config)
    local = LlamaAttention(local_cfg).eval().float()
    _load_attn_local(local, hf_attn, local_cfg)
    local = local.to("nntile")

    b, s, h = 2, 8, tiny_hf_config.hidden_size
    x = torch.randn(b, s, h)
    position_ids = torch.arange(s).unsqueeze(0).expand(b, s)
    from torch_nntile.rope import rope_sin_cos_from_position_ids

    sin, cos = rope_sin_cos_from_position_ids(
        position_ids, local_cfg.head_dim, rope_theta=local_cfg.rope_theta
    )
    pos_emb = _hf_position_embeddings(tiny_hf_config, position_ids)
    # Causal mask for HF eager SDPA path (4-D additive).
    causal = torch.zeros(b, 1, s, s)
    causal.masked_fill_(
        torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    with torch.no_grad():
        ref = hf_attn(
            x,
            position_embeddings=pos_emb,
            attention_mask=causal,
        )[0]
        out = local(
            contiguous_to_nntile(x),
            sin=sin,
            cos=cos,
            is_causal=True,
        )
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


def test_llama_decoder_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(4)
    hf_layer = HfDecoder(tiny_hf_config, layer_idx=0).eval().float()
    local_cfg = llama_config_from_hf(tiny_hf_config)
    local = LlamaDecoder(local_cfg).eval().float()
    local.input_layernorm.weight.data.copy_(
        hf_layer.input_layernorm.weight.data
    )
    local.post_attention_layernorm.weight.data.copy_(
        hf_layer.post_attention_layernorm.weight.data
    )
    _load_attn_local(local.self_attn, hf_layer.self_attn, local_cfg)
    copy_linear(local.mlp.gate_proj, hf_layer.mlp.gate_proj)
    copy_linear(local.mlp.up_proj, hf_layer.mlp.up_proj)
    copy_linear(local.mlp.down_proj, hf_layer.mlp.down_proj)
    local = local.to("nntile")

    b, s, h = 2, 8, tiny_hf_config.hidden_size
    x = torch.randn(b, s, h)
    position_ids = torch.arange(s).unsqueeze(0).expand(b, s)
    from torch_nntile.rope import rope_sin_cos_from_position_ids

    sin, cos = rope_sin_cos_from_position_ids(
        position_ids, local_cfg.head_dim, rope_theta=local_cfg.rope_theta
    )
    pos_emb = _hf_position_embeddings(tiny_hf_config, position_ids)
    causal = torch.zeros(b, 1, s, s)
    causal.masked_fill_(
        torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    with torch.no_grad():
        ref = hf_layer(
            x,
            attention_mask=causal,
            position_ids=position_ids,
            position_embeddings=pos_emb,
        )[0]
        out = local(
            contiguous_to_nntile(x),
            sin=sin,
            cos=cos,
            is_causal=True,
        )
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


def test_llama_causal_forward_matches_hf(tiny_hf_config):
    hf, minimal = _make_models(tiny_hf_config)
    input_ids = torch.randint(0, tiny_hf_config.vocab_size, (2, 8))
    with torch.no_grad():
        ref = hf(input_ids).logits
        out = minimal(contiguous_to_nntile(input_ids))
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


def test_llama_causal_gqa_forward_matches_hf(tiny_gqa_hf_config):
    hf, minimal = _make_models(tiny_gqa_hf_config)
    input_ids = torch.randint(0, tiny_gqa_hf_config.vocab_size, (2, 8))
    with torch.no_grad():
        ref = hf(input_ids).logits
        out = minimal(contiguous_to_nntile(input_ids))
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


def test_llama_causal_backward_matches_hf(tiny_hf_config):
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
        minimal.model.embed_tokens.weight,
        grad_outputs=contiguous_to_nntile(grad),
    )
    assert_close(gw, hf.model.embed_tokens.weight.grad, rtol=1e-3, atol=1e-3)
