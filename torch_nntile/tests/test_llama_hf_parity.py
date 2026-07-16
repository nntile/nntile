# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_llama_hf_parity.py
# Thorough Llama submodule parity vs HuggingFace (mirrors deleted NNGraph
# llama_{config,mlp,attention,rope,decoder,model,causal} matrix).

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

from torch_nntile import _C
from torch_nntile.models.hf_rope_layout import (
    copy_linear,
    hf_to_nntile_qkv_weight,
)
from torch_nntile.models.llama import (
    LlamaAttention,
    LlamaCausal,
    LlamaConfig,
    LlamaDecoder,
    LlamaMLP,
    LlamaRMSNorm,
)
from torch_nntile.models.llama_hf_loader import (
    llama_config_from_hf,
    load_hf_into_llama_causal,
)
from torch_nntile.nn.linear import (
    linear_to_gqa_output_weight,
    linear_to_gqa_q_weight,
    linear_to_output_weight,
    linear_to_qkv_weight,
)
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
BWD_ATOL = 1e-3


# ---------------------------------------------------------------------------
# Config (was llama_config.cc)
# ---------------------------------------------------------------------------


def test_llama_config_defaults_and_validate():
    cfg = LlamaConfig()
    assert cfg.head_dim == cfg.hidden_size // cfg.num_attention_heads
    cfg.validate()
    bad = LlamaConfig(hidden_size=65, num_attention_heads=4)
    with pytest.raises(ValueError):
        bad.validate()
    bad_gqa = LlamaConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=3,
    )
    with pytest.raises(ValueError):
        bad_gqa.validate()


def test_llama_config_from_hf_roundtrip():
    hf = HfLlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
    )
    local = llama_config_from_hf(hf)
    assert local.num_key_value_heads == 2
    assert local.head_dim == 16


# ---------------------------------------------------------------------------
# Tiny HF fixtures
# ---------------------------------------------------------------------------


def _hf_cfg(*, gqa: bool = False) -> HfLlamaConfig:
    cfg = HfLlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2 if gqa else 4,
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


def _hf_pos_emb(hf_cfg: HfLlamaConfig, position_ids: torch.Tensor):
    rotary = LlamaRotaryEmbedding(config=hf_cfg)
    hidden = torch.zeros(
        position_ids.shape[0],
        position_ids.shape[1],
        hf_cfg.hidden_size,
        dtype=torch.float32,
    )
    return rotary(hidden, position_ids)


def _load_attn(local: LlamaAttention, hf_attn, cfg: LlamaConfig) -> None:
    q_weight = hf_to_nntile_qkv_weight(
        hf_attn.q_proj.weight.data,
        n_heads=cfg.num_attention_heads,
        head_dim=cfg.head_dim,
    )
    if local.use_gqa:
        local.q_weight.data.copy_(
            linear_to_gqa_q_weight(
                q_weight,
                n_kv_heads=cfg.num_key_value_heads,
                n_rep=local.n_rep,
                head_size=cfg.head_dim,
            )
        )
    else:
        local.q_weight.data.copy_(
            linear_to_qkv_weight(
                q_weight,
                n_heads=cfg.num_attention_heads,
                head_size=cfg.head_dim,
            )
        )
    k_weight = hf_to_nntile_qkv_weight(
        hf_attn.k_proj.weight.data,
        n_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )
    local.k_weight.data.copy_(
        linear_to_qkv_weight(
            k_weight,
            n_heads=cfg.num_key_value_heads,
            head_size=cfg.head_dim,
        )
    )
    local.v_weight.data.copy_(
        linear_to_qkv_weight(
            hf_attn.v_proj.weight.data,
            n_heads=cfg.num_key_value_heads,
            head_size=cfg.head_dim,
        )
    )
    if local.use_gqa:
        local.o_weight.data.copy_(
            linear_to_gqa_output_weight(
                hf_attn.o_proj.weight.data,
                n_kv_heads=cfg.num_key_value_heads,
                n_rep=local.n_rep,
                head_size=cfg.head_dim,
            )
        )
    else:
        local.o_weight.data.copy_(
            linear_to_output_weight(
                hf_attn.o_proj.weight.data,
                n_heads=cfg.num_attention_heads,
                head_size=cfg.head_dim,
            )
        )


def _make_causal(hf_cfg: HfLlamaConfig):
    torch.manual_seed(0)
    hf = LlamaForCausalLM(hf_cfg).eval().float()
    local = LlamaCausal(llama_config_from_hf(hf_cfg)).eval().float()
    load_hf_into_llama_causal(local, hf)
    return hf, local.to("nntile")


# ---------------------------------------------------------------------------
# MLP (was llama_mlp.cc)
# ---------------------------------------------------------------------------


def test_llama_mlp_forward_backward_matches_hf():
    torch.manual_seed(1)
    hf_cfg = _hf_cfg()
    hf_mlp = HfMLP(hf_cfg).eval().float()
    local = LlamaMLP(llama_config_from_hf(hf_cfg)).eval().float()
    copy_linear(local.gate_proj, hf_mlp.gate_proj)
    copy_linear(local.up_proj, hf_mlp.up_proj)
    copy_linear(local.down_proj, hf_mlp.down_proj)
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
        (x_n, local_n.gate_proj.weight),
        contiguous_to_nntile(grad),
    )
    assert_close(gx, x.grad, rtol=1e-3, atol=BWD_ATOL)
    assert_close(gw, hf_mlp.gate_proj.weight.grad, rtol=1e-3, atol=BWD_ATOL)


# ---------------------------------------------------------------------------
# RoPE (was llama_rope.cc)
# ---------------------------------------------------------------------------


def test_llama_rope_sin_cos_matches_hf_half_channels():
    torch.manual_seed(2)
    hf_cfg = _hf_cfg()
    b, s, hd = 2, 8, hf_cfg.hidden_size // hf_cfg.num_attention_heads
    position_ids = torch.arange(s).unsqueeze(0).expand(b, s)
    cos_hf, sin_hf = _hf_pos_emb(hf_cfg, position_ids)
    # HF returns full head_dim with duplicated halves.
    half = hd // 2
    sin_local, cos_local = rope_sin_cos_from_position_ids(
        position_ids, hd, rope_theta=hf_cfg.rope_theta
    )
    assert_close(sin_local, sin_hf[..., :half], rtol=1e-5, atol=1e-5)
    assert_close(cos_local, cos_hf[..., :half], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Attention matrix: MHA/GQA x RoPE/no-RoPE x causal/no-mask
# (was llama_attention.cc)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("use_rope", [True, False])
@pytest.mark.parametrize("causal", [True, False])
def test_llama_attention_forward_backward_matrix(gqa, use_rope, causal):
    torch.manual_seed(10 + int(gqa) * 4 + int(use_rope) * 2 + int(causal))
    hf_cfg = _hf_cfg(gqa=gqa)
    hf_attn = HfAttention(hf_cfg, layer_idx=0).eval().float()
    local_cfg = llama_config_from_hf(hf_cfg)
    local = LlamaAttention(local_cfg).eval().float()
    _load_attn(local, hf_attn, local_cfg)
    local_n = local.to("nntile")

    b, s, h = 2, 8, hf_cfg.hidden_size
    x = torch.randn(b, s, h, requires_grad=True)
    position_ids = torch.arange(s).unsqueeze(0).expand(b, s)
    pos_emb = _hf_pos_emb(hf_cfg, position_ids)
    if use_rope:
        sin, cos = rope_sin_cos_from_position_ids(
            position_ids, local_cfg.head_dim, rope_theta=local_cfg.rope_theta
        )
    else:
        half = local_cfg.head_dim // 2
        sin = torch.zeros(b, s, half)
        cos = torch.ones(b, s, half)
        pos_emb = (torch.ones_like(pos_emb[0]), torch.zeros_like(pos_emb[1]))

    # Prefer ``is_causal=True`` over additive float masks - nntile SDPA rejects
    # common additive ``-inf`` mask layouts (``view: storage alias``).
    y_ref = hf_attn(
        x,
        position_embeddings=pos_emb,
        attention_mask=additive_causal_mask(b, s) if causal else None,
    )[0]
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)

    x_n = contiguous_to_nntile(x.detach()).requires_grad_(True)
    y = local_n(
        x_n,
        sin=contiguous_to_nntile(sin),
        cos=contiguous_to_nntile(cos),
        attn_mask=None,
        is_causal=causal,
    )
    assert_close(y, y_ref.detach(), rtol=RTOL, atol=5e-4)
    (gx,) = torch.autograd.grad(y, x_n, contiguous_to_nntile(grad))
    assert_close(gx, x.grad, rtol=1e-3, atol=5e-3)


# ---------------------------------------------------------------------------
# Decoder (was llama_decoder.cc)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gqa", [False, True])
def test_llama_decoder_forward_backward_matches_hf(gqa):
    torch.manual_seed(20 + int(gqa))
    hf_cfg = _hf_cfg(gqa=gqa)
    hf_layer = HfDecoder(hf_cfg, layer_idx=0).eval().float()
    local_cfg = llama_config_from_hf(hf_cfg)
    local = LlamaDecoder(local_cfg).eval().float()
    local.input_layernorm.weight.data.copy_(
        hf_layer.input_layernorm.weight.data
    )
    local.post_attention_layernorm.weight.data.copy_(
        hf_layer.post_attention_layernorm.weight.data
    )
    _load_attn(local.self_attn, hf_layer.self_attn, local_cfg)
    copy_linear(local.mlp.gate_proj, hf_layer.mlp.gate_proj)
    copy_linear(local.mlp.up_proj, hf_layer.mlp.up_proj)
    copy_linear(local.mlp.down_proj, hf_layer.mlp.down_proj)
    local_n = local.to("nntile")

    b, s, h = 2, 8, hf_cfg.hidden_size
    x = torch.randn(b, s, h, requires_grad=True)
    position_ids = torch.arange(s).unsqueeze(0).expand(b, s)
    sin, cos = rope_sin_cos_from_position_ids(
        position_ids, local_cfg.head_dim, rope_theta=local_cfg.rope_theta
    )
    pos_emb = _hf_pos_emb(hf_cfg, position_ids)
    causal = additive_causal_mask(b, s)
    y_ref = hf_layer(
        x,
        attention_mask=causal,
        position_ids=position_ids,
        position_embeddings=pos_emb,
    )[0]
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)

    x_n = contiguous_to_nntile(x.detach()).requires_grad_(True)
    y = local_n(
        x_n,
        sin=contiguous_to_nntile(sin),
        cos=contiguous_to_nntile(cos),
        attn_mask=None,
        is_causal=True,
    )
    assert_close(y, y_ref.detach(), rtol=RTOL, atol=5e-4)
    (gx,) = torch.autograd.grad(y, x_n, contiguous_to_nntile(grad))
    assert_close(gx, x.grad, rtol=1e-3, atol=5e-3)


# ---------------------------------------------------------------------------
# Model + Causal (was llama_model.cc / llama_causal.cc)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gqa", [False, True])
def test_llama_model_forward_matches_hf(gqa):
    hf_cfg = _hf_cfg(gqa=gqa)
    hf, minimal = _make_causal(hf_cfg)
    ids = torch.randint(0, hf_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        ref = hf.model(ids).last_hidden_state
        out = minimal.model(contiguous_to_nntile(ids))
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


@pytest.mark.parametrize("gqa", [False, True])
def test_llama_causal_forward_matches_hf(gqa):
    hf_cfg = _hf_cfg(gqa=gqa)
    assert hf_cfg.tie_word_embeddings is False
    hf, minimal = _make_causal(hf_cfg)
    ids = torch.randint(0, hf_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        ref = hf(ids).logits
        out = minimal(contiguous_to_nntile(ids))
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


def test_llama_causal_backward_matches_hf():
    hf_cfg = _hf_cfg()
    assert hf_cfg.tie_word_embeddings is False
    hf, minimal = _make_causal(hf_cfg)
    for p in hf.parameters():
        p.requires_grad_(True)
    for p in minimal.parameters():
        p.requires_grad_(True)
    ids = torch.randint(0, hf_cfg.vocab_size, (2, 8))
    grad = torch.randn(2, 8, hf_cfg.vocab_size)
    hf(ids).logits.backward(grad)
    logits = minimal(contiguous_to_nntile(ids))
    gw_emb, gw_lm = torch.autograd.grad(
        logits,
        (minimal.model.embed_tokens.weight, minimal.lm_head.weight),
        contiguous_to_nntile(grad),
    )
    assert_close(
        gw_emb, hf.model.embed_tokens.weight.grad, rtol=1e-3, atol=BWD_ATOL
    )
    assert_close(gw_lm, hf.lm_head.weight.grad, rtol=1e-3, atol=BWD_ATOL)


def test_llama_rms_norm_module_matches_cpu():
    torch.manual_seed(3)
    mod = LlamaRMSNorm(64, eps=1e-6).float()
    with torch.no_grad():
        mod.weight.normal_(mean=1.0, std=0.05)
    x = torch.randn(2, 8, 64)
    y_ref = mod(x)
    y = mod.to("nntile")(contiguous_to_nntile(x))
    assert_close(y, y_ref, rtol=RTOL, atol=ATOL)
