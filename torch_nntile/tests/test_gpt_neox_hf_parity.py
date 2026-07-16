# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_gpt_neox_hf_parity.py
# Thorough GPT-NeoX submodule parity vs HuggingFace (mirrors deleted NNGraph
# gpt_neox_{config,mlp,attention,decoder,model,causal} matrix).

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
from torch_nntile.nn.linear import linear_to_output_weight
from torch_nntile.rope import rope_sin_cos_from_position_ids
from parity_helpers import (
    additive_causal_mask,
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
# Config (was gpt_neox_config.cc)
# ---------------------------------------------------------------------------


def test_gpt_neox_config_defaults_validate_and_rotary_ndims():
    cfg = GPTNeoXConfig(hidden_size=64, num_attention_heads=4, rotary_pct=0.25)
    assert cfg.head_dim == 16
    assert cfg.rotary_ndims == 4
    cfg.validate()

    no_rope = GPTNeoXConfig(
        hidden_size=64,
        num_attention_heads=4,
        rotary_pct=0.0,
    )
    assert no_rope.rotary_ndims == 0
    no_rope.validate()

    bad = GPTNeoXConfig(hidden_size=66, num_attention_heads=4)
    with pytest.raises(ValueError):
        bad.validate()


def test_gpt_neox_config_from_hf_preserves_rotary_ndims():
    hf = _hf_cfg(rotary_pct=0.5)
    local = gpt_neox_config_from_hf(hf)
    assert local.head_dim == 16
    assert local.rotary_pct == 0.5
    assert local.rotary_ndims == 8
    local.validate()


# ---------------------------------------------------------------------------
# Tiny HF fixtures
# ---------------------------------------------------------------------------


def _hf_cfg(
    *,
    rotary_pct: float = 0.25,
    attention_bias: bool = True,
) -> HfGPTNeoXConfig:
    cfg = HfGPTNeoXConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        layer_norm_eps=1e-5,
        rotary_pct=rotary_pct,
        rotary_emb_base=10000.0,
        use_parallel_residual=True,
        attention_bias=attention_bias,
        tie_word_embeddings=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        hidden_act="gelu",
    )
    cfg._attn_implementation = "eager"
    return cfg


def _make_causal(hf_cfg: HfGPTNeoXConfig):
    torch.manual_seed(0)
    hf = GPTNeoXForCausalLM(hf_cfg).eval().float()
    local = GPTNeoXCausal(gpt_neox_config_from_hf(hf_cfg)).eval().float()
    load_hf_into_gpt_neox_causal(local, hf)
    return hf, local.to("nntile")


def _hf_position_embeddings(hf_cfg: HfGPTNeoXConfig, position_ids):
    rotary = GPTNeoXRotaryEmbedding(config=hf_cfg)
    hidden = torch.zeros(
        position_ids.shape[0],
        position_ids.shape[1],
        hf_cfg.hidden_size,
        dtype=torch.float32,
    )
    return rotary(hidden, position_ids)


def _identity_position_embeddings(hf_cfg: HfGPTNeoXConfig, position_ids):
    cos, sin = _hf_position_embeddings(hf_cfg, position_ids)
    return torch.ones_like(cos), torch.zeros_like(sin)


def _local_sin_cos(local_cfg: GPTNeoXConfig, position_ids):
    if local_cfg.rotary_ndims <= 0:
        return None, None
    return rope_sin_cos_from_position_ids(
        position_ids,
        local_cfg.rotary_ndims,
        rope_theta=local_cfg.rotary_emb_base,
    )


def _load_attn(local: GPTNeoXAttention, hf_attn, cfg: GPTNeoXConfig) -> None:
    pct = cfg.rotary_ndims / cfg.head_dim if cfg.head_dim > 0 else 0.0
    fused = hf_to_nntile_fused_qkv_weight(
        hf_attn.query_key_value.weight.data,
        n_heads=cfg.num_attention_heads,
        head_dim=cfg.head_dim,
        rotary_pct=pct,
    )
    fused = fused.reshape(cfg.num_attention_heads, 3 * cfg.head_dim, -1)
    local.q_weight.data.copy_(
        fused[:, : cfg.head_dim, :].permute(2, 1, 0).contiguous()
    )
    local.k_weight.data.copy_(
        fused[:, cfg.head_dim : 2 * cfg.head_dim, :]
        .permute(2, 1, 0)
        .contiguous()
    )
    local.v_weight.data.copy_(
        fused[:, 2 * cfg.head_dim : 3 * cfg.head_dim, :]
        .permute(2, 1, 0)
        .contiguous()
    )
    local.o_weight.data.copy_(
        linear_to_output_weight(
            hf_attn.dense.weight.data,
            n_heads=cfg.num_attention_heads,
            head_size=cfg.head_dim,
        )
    )
    if local.q_bias is not None and hf_attn.query_key_value.bias is not None:
        fused_b = hf_to_nntile_fused_qkv_bias(
            hf_attn.query_key_value.bias.data,
            n_heads=cfg.num_attention_heads,
            head_dim=cfg.head_dim,
            rotary_pct=pct,
        ).reshape(cfg.num_attention_heads, 3 * cfg.head_dim)
        local.q_bias.data.copy_(fused_b[:, : cfg.head_dim])
        local.k_bias.data.copy_(
            fused_b[:, cfg.head_dim : 2 * cfg.head_dim]
        )
        local.v_bias.data.copy_(
            fused_b[:, 2 * cfg.head_dim : 3 * cfg.head_dim]
        )
    if local.o_bias is not None and hf_attn.dense.bias is not None:
        local.o_bias.data.copy_(hf_attn.dense.bias.data)


def _load_layer(local: GPTNeoXLayer, hf_layer: HfLayer, cfg: GPTNeoXConfig):
    local.input_layernorm.load_state_dict(hf_layer.input_layernorm.state_dict())
    local.post_attention_layernorm.load_state_dict(
        hf_layer.post_attention_layernorm.state_dict()
    )
    _load_attn(local.attention, hf_layer.attention, cfg)
    copy_linear(local.mlp.dense_h_to_4h, hf_layer.mlp.dense_h_to_4h)
    copy_linear(local.mlp.dense_4h_to_h, hf_layer.mlp.dense_4h_to_h)


# ---------------------------------------------------------------------------
# MLP (was gpt_neox_mlp.cc)
# ---------------------------------------------------------------------------


def test_gpt_neox_mlp_forward_backward_matches_hf():
    torch.manual_seed(1)
    hf_cfg = _hf_cfg()
    hf_mlp = HfMLP(hf_cfg).eval().float()
    local = GPTNeoXMLP(gpt_neox_config_from_hf(hf_cfg)).eval().float()
    copy_linear(local.dense_h_to_4h, hf_mlp.dense_h_to_4h)
    copy_linear(local.dense_4h_to_h, hf_mlp.dense_4h_to_h)
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
        (x_n, local_n.dense_h_to_4h.weight),
        contiguous_to_nntile(grad),
    )
    assert_close(gx, x.grad, rtol=1e-3, atol=BWD_ATOL)
    assert_close(
        gw, hf_mlp.dense_h_to_4h.weight.grad, rtol=1e-3, atol=BWD_ATOL
    )


# ---------------------------------------------------------------------------
# Attention matrix: RoPE/no-RoPE x causal/no-mask x bias/no-bias
# (was gpt_neox_attention.cc)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_rope", [True, False])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("attention_bias", [True, False])
def test_gpt_neox_attention_forward_backward_matrix(
    use_rope,
    causal,
    attention_bias,
):
    seed = 10 + int(use_rope) * 4 + int(causal) * 2 + int(attention_bias)
    torch.manual_seed(seed)
    hf_cfg = _hf_cfg(attention_bias=attention_bias)
    hf_attn = HfAttention(hf_cfg, layer_idx=0).eval().float()
    local_cfg = gpt_neox_config_from_hf(hf_cfg)
    local = GPTNeoXAttention(local_cfg).eval().float()
    _load_attn(local, hf_attn, local_cfg)
    local_n = local.to("nntile")

    b, s, h = 2, 8, hf_cfg.hidden_size
    x = torch.randn(b, s, h, requires_grad=True)
    position_ids = torch.arange(s).unsqueeze(0).expand(b, s)
    if use_rope:
        pos_emb = _hf_position_embeddings(hf_cfg, position_ids)
        sin, cos = _local_sin_cos(local_cfg, position_ids)
    else:
        pos_emb = _identity_position_embeddings(hf_cfg, position_ids)
        sin, cos = None, None
    mask = additive_causal_mask(b, s) if causal else None

    y_ref = hf_attn(
        x,
        attention_mask=mask,
        position_embeddings=pos_emb,
        cache_position=torch.arange(s),
    )[0]

    x_n = contiguous_to_nntile(x.detach()).requires_grad_(True)
    y = local_n(
        x_n,
        sin=None if sin is None else contiguous_to_nntile(sin),
        cos=None if cos is None else contiguous_to_nntile(cos),
        attn_mask=None,
        is_causal=causal,
    )
    assert_close(y, y_ref.detach(), rtol=RTOL, atol=ATTN_ATOL)

    if use_rope:
        return

    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    (gx,) = torch.autograd.grad(y, x_n, contiguous_to_nntile(grad))
    assert_close(gx, x.grad, rtol=1e-3, atol=BWD_ATOL)


# ---------------------------------------------------------------------------
# Decoder layer (was gpt_neox_decoder.cc)
# ---------------------------------------------------------------------------


def test_gpt_neox_layer_forward_matches_hf():
    torch.manual_seed(20)
    hf_cfg = _hf_cfg()
    hf_layer = HfLayer(hf_cfg, layer_idx=0).eval().float()
    local_cfg = gpt_neox_config_from_hf(hf_cfg)
    local = GPTNeoXLayer(local_cfg).eval().float()
    _load_layer(local, hf_layer, local_cfg)
    local_n = local.to("nntile")

    b, s, h = 2, 8, hf_cfg.hidden_size
    x = torch.randn(b, s, h)
    position_ids = torch.arange(s).unsqueeze(0).expand(b, s)
    sin, cos = _local_sin_cos(local_cfg, position_ids)
    mask = additive_causal_mask(b, s)
    with torch.no_grad():
        ref = hf_layer(
            x,
            attention_mask=mask,
            position_ids=position_ids,
            position_embeddings=_hf_position_embeddings(hf_cfg, position_ids),
            cache_position=torch.arange(s),
        )[0]
        out = local_n(
            contiguous_to_nntile(x),
            sin=contiguous_to_nntile(sin),
            cos=contiguous_to_nntile(cos),
            attn_mask=None,
            is_causal=True,
        )
    assert_close(out, ref, rtol=RTOL, atol=ATTN_ATOL)


# ---------------------------------------------------------------------------
# Model + Causal LM (was gpt_neox_model.cc / gpt_neox_causal.cc)
# ---------------------------------------------------------------------------


def test_gpt_neox_model_hidden_forward_matches_hf():
    hf_cfg = _hf_cfg()
    hf, local = _make_causal(hf_cfg)
    ids = torch.randint(0, hf_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        ref = hf.gpt_neox(ids).last_hidden_state
        out = local.gpt_neox(contiguous_to_nntile(ids))
    assert_close(out, ref, rtol=RTOL, atol=ATTN_ATOL)


def test_gpt_neox_causal_forward_matches_hf():
    hf_cfg = _hf_cfg()
    hf, local = _make_causal(hf_cfg)
    ids = torch.randint(0, hf_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        ref = hf(ids).logits
        out = local(contiguous_to_nntile(ids))
    assert_close(out, ref, rtol=RTOL, atol=ATTN_ATOL)


def test_gpt_neox_causal_backward_matches_hf():
    hf_cfg = _hf_cfg()
    assert hf_cfg.tie_word_embeddings is False
    hf, local = _make_causal(hf_cfg)
    assert local.embed_out.weight is not local.gpt_neox.embed_in.weight
    for p in hf.parameters():
        p.requires_grad_(True)
    for p in local.parameters():
        p.requires_grad_(True)

    ids = torch.randint(0, hf_cfg.vocab_size, (2, 8))
    grad = torch.randn(2, 8, hf_cfg.vocab_size)
    hf(ids).logits.backward(grad)
    logits = local(contiguous_to_nntile(ids))
    # Only lm-head weight: full-graph bwd through partial RoPE ``narrow`` is
    # not implemented on nntile yet.
    (gw_out,) = torch.autograd.grad(
        logits,
        local.embed_out.weight,
        contiguous_to_nntile(grad),
    )
    assert_close(gw_out, hf.embed_out.weight.grad, rtol=1e-3, atol=BWD_ATOL)
