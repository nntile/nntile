# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_gpt_neox_hf_parity.py
# GPT-NeoX layer + full-model parity vs HuggingFace GPTNeoXForCausalLM.

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("transformers")

import torch
from transformers import GPTNeoXConfig as HfGPTNeoXConfig
from transformers import GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention as HfAttention,
    GPTNeoXLayer as HfLayer,
    GPTNeoXMLP as HfMLP,
    GPTNeoXRotaryEmbedding,
)

import torch_nntile
from torch_nntile import _C
from torch_nntile.models.gpt_neox import (
    GPTNeoXAttention,
    GPTNeoXCausal,
    GPTNeoXConfig,
    GPTNeoXLayer,
    GPTNeoXMLP,
)
from torch_nntile.models.gpt_neox_hf_loader import (
    gpt_neox_config_from_hf,
    load_hf_into_gpt_neox_causal,
)
from torch_nntile.models.hf_rope_layout import (
    copy_linear,
    hf_to_nntile_fused_qkv_bias,
    hf_to_nntile_fused_qkv_weight,
)
from torch_nntile.rope import rope_sin_cos_from_position_ids
from parity_helpers import assert_close, contiguous_to_nntile


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)

RTOL = 1e-4
ATOL = 1e-4


@pytest.fixture
def tiny_hf_config() -> HfGPTNeoXConfig:
    cfg = HfGPTNeoXConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        layer_norm_eps=1e-5,
        rotary_pct=0.25,
        rotary_emb_base=10000.0,
        use_parallel_residual=True,
        attention_bias=True,
        tie_word_embeddings=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        hidden_act="gelu",
    )
    cfg._attn_implementation = "eager"
    return cfg


def _make_models(hf_cfg: HfGPTNeoXConfig):
    torch.manual_seed(0)
    hf = GPTNeoXForCausalLM(hf_cfg).eval().float()
    local_cfg = gpt_neox_config_from_hf(hf_cfg)
    minimal = GPTNeoXCausal(local_cfg).eval().float()
    load_hf_into_gpt_neox_causal(minimal, hf)
    minimal = minimal.to("nntile")
    return hf, minimal


def _load_attn(local: GPTNeoXAttention, hf_attn, cfg: GPTNeoXConfig) -> None:
    local.query_key_value.weight.data.copy_(
        hf_to_nntile_fused_qkv_weight(
            hf_attn.query_key_value.weight.data,
            n_heads=cfg.num_attention_heads,
            head_dim=cfg.head_dim,
            rotary_pct=cfg.rotary_pct,
        )
    )
    copy_linear(local.dense, hf_attn.dense)
    if local.query_key_value.bias is not None:
        local.query_key_value.bias.data.copy_(
            hf_to_nntile_fused_qkv_bias(
                hf_attn.query_key_value.bias.data,
                n_heads=cfg.num_attention_heads,
                head_dim=cfg.head_dim,
                rotary_pct=cfg.rotary_pct,
            )
        )


def test_gpt_neox_mlp_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(1)
    hf_mlp = HfMLP(tiny_hf_config).eval().float()
    local = GPTNeoXMLP(gpt_neox_config_from_hf(tiny_hf_config)).eval().float()
    copy_linear(local.dense_h_to_4h, hf_mlp.dense_h_to_4h)
    copy_linear(local.dense_4h_to_h, hf_mlp.dense_4h_to_h)
    local = local.to("nntile")
    x = torch.randn(2, 8, tiny_hf_config.hidden_size)
    with torch.no_grad():
        ref = hf_mlp(x)
        out = local(contiguous_to_nntile(x))
    assert_close(out, ref, rtol=RTOL, atol=ATOL)


def _hf_position_embeddings(hf_cfg: HfGPTNeoXConfig, position_ids: torch.Tensor):
    rotary = GPTNeoXRotaryEmbedding(config=hf_cfg)
    hidden = torch.zeros(
        position_ids.shape[0],
        position_ids.shape[1],
        hf_cfg.hidden_size,
        dtype=torch.float32,
    )
    return rotary(hidden, position_ids)


def _causal_mask(batch: int, seq: int) -> torch.Tensor:
    mask = torch.zeros(batch, 1, seq, seq)
    mask.masked_fill_(
        torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return mask


def test_gpt_neox_attention_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(2)
    hf_attn = HfAttention(tiny_hf_config).eval().float()
    local_cfg = gpt_neox_config_from_hf(tiny_hf_config)
    local = GPTNeoXAttention(local_cfg).eval().float()
    _load_attn(local, hf_attn, local_cfg)
    local = local.to("nntile")

    b, s, h = 2, 8, tiny_hf_config.hidden_size
    x = torch.randn(b, s, h)
    position_ids = torch.arange(s).unsqueeze(0).expand(b, s)
    sin, cos = rope_sin_cos_from_position_ids(
        position_ids,
        local_cfg.rotary_ndims,
        rope_theta=local_cfg.rotary_emb_base,
    )
    pos_emb = _hf_position_embeddings(tiny_hf_config, position_ids)
    causal = _causal_mask(b, s)
    with torch.no_grad():
        ref = hf_attn(
            x,
            attention_mask=causal,
            position_embeddings=pos_emb,
        )[0]
        out = local(
            contiguous_to_nntile(x),
            sin=sin,
            cos=cos,
            is_causal=True,
        )
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


def test_gpt_neox_layer_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(3)
    hf_layer = HfLayer(tiny_hf_config).eval().float()
    local_cfg = gpt_neox_config_from_hf(tiny_hf_config)
    local = GPTNeoXLayer(local_cfg).eval().float()
    local.input_layernorm.load_state_dict(hf_layer.input_layernorm.state_dict())
    local.post_attention_layernorm.load_state_dict(
        hf_layer.post_attention_layernorm.state_dict()
    )
    _load_attn(local.attention, hf_layer.attention, local_cfg)
    copy_linear(local.mlp.dense_h_to_4h, hf_layer.mlp.dense_h_to_4h)
    copy_linear(local.mlp.dense_4h_to_h, hf_layer.mlp.dense_4h_to_h)
    local = local.to("nntile")

    b, s, h = 2, 8, tiny_hf_config.hidden_size
    x = torch.randn(b, s, h)
    position_ids = torch.arange(s).unsqueeze(0).expand(b, s)
    sin, cos = rope_sin_cos_from_position_ids(
        position_ids,
        local_cfg.rotary_ndims,
        rope_theta=local_cfg.rotary_emb_base,
    )
    pos_emb = _hf_position_embeddings(tiny_hf_config, position_ids)
    causal = _causal_mask(b, s)
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


def test_gpt_neox_causal_forward_matches_hf(tiny_hf_config):
    hf, minimal = _make_models(tiny_hf_config)
    input_ids = torch.randint(0, tiny_hf_config.vocab_size, (2, 8))
    with torch.no_grad():
        ref = hf(input_ids).logits
        out = minimal(contiguous_to_nntile(input_ids))
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


def test_gpt_neox_causal_backward_matches_hf(tiny_hf_config):
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
        minimal.gpt_neox.embed_in.weight,
        grad_outputs=contiguous_to_nntile(grad),
    )
    assert_close(gw, hf.gpt_neox.embed_in.weight.grad, rtol=1e-3, atol=1e-3)
