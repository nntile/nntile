# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_t5_hf_parity.py
# Thorough T5 submodule parity vs HuggingFace (mirrors deleted NNGraph
# t5_{config,ff,attention,encoder,decoder,model,conditional} matrix).

from __future__ import annotations

import copy

import pytest

pytest.importorskip("numpy")
pytest.importorskip("transformers")

import torch
from conftest import nntile_cpu
from parity_helpers import (
    additive_causal_mask,
    assert_close,
    contiguous_to_nntile,
)
from torch_nntile.models.hf_rope_layout import copy_linear
from torch_nntile.models.t5 import (
    T5Attention,
    T5Config,
    T5DecoderBlock,
    T5EncoderBlock,
    T5ForConditionalGeneration,
    T5LayerFF,
)
from torch_nntile.models.t5_hf_loader import (
    disable_t5_relative_attention_bias,
    load_hf_into_t5,
    t5_config_from_hf,
)
from torch_nntile.nn.linear import (
    linear_to_output_weight,
    linear_to_qkv_weight,
    qkv_to_linear_weight,
)
from transformers import T5Config as HfT5Config
from transformers import T5ForConditionalGeneration as HfT5
from transformers.models.t5.modeling_t5 import (
    T5Attention as HfAttention,
)
from transformers.models.t5.modeling_t5 import (
    T5Block as HfBlock,
)
from transformers.models.t5.modeling_t5 import (
    T5LayerFF as HfLayerFF,
)

RTOL = 1e-4
ATOL = 1e-4
ATTN_ATOL = 5e-4
BWD_ATOL = 1e-3

# ---------------------------------------------------------------------------
# Config (was t5_config.cc)
# ---------------------------------------------------------------------------


def test_t5_config_validate_and_head_dim():
    cfg = T5Config(d_model=64, d_kv=16, num_heads=4)
    assert cfg.head_dim == 16
    assert cfg.inner_dim == 64
    cfg.validate()

    bad = T5Config(d_model=0, d_kv=16, num_heads=4)
    with pytest.raises(ValueError):
        bad.validate()


def test_t5_config_from_hf_preserves_head_dim_and_tie_flag():
    hf = _hf_cfg()
    local = t5_config_from_hf(hf)
    assert local.head_dim == hf.d_kv
    assert local.inner_dim == hf.num_heads * hf.d_kv
    assert local.tie_word_embeddings is False
    local.validate()


# ---------------------------------------------------------------------------
# Tiny HF fixtures
# ---------------------------------------------------------------------------


def _hf_cfg() -> HfT5Config:
    cfg = HfT5Config(
        vocab_size=128,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        relative_attention_num_buckets=32,
        layer_norm_epsilon=1e-6,
        dropout_rate=0.0,
        feed_forward_proj="gated-gelu",
        is_gated_act=True,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _decoder_hf_cfg(hf_cfg: HfT5Config) -> HfT5Config:
    cfg = copy.deepcopy(hf_cfg)
    cfg.is_decoder = True
    cfg.is_encoder_decoder = False
    return cfg


def _make_models(hf_cfg: HfT5Config):
    torch.manual_seed(0)
    hf = HfT5(hf_cfg).eval().float()
    disable_t5_relative_attention_bias(hf)
    local_cfg = t5_config_from_hf(hf_cfg)
    local_cfg.tie_word_embeddings = False
    local = T5ForConditionalGeneration(local_cfg).eval().float()
    load_hf_into_t5(local, hf)
    return hf, local.to("nntile")


def _load_attn(local: T5Attention, hf_attn: HfAttention) -> None:
    local.q_weight.data.copy_(
        linear_to_qkv_weight(
            hf_attn.q.weight.data,
            n_heads=local.n_heads,
            head_size=local.key_value_proj_dim,
        )
    )
    local.k_weight.data.copy_(
        linear_to_qkv_weight(
            hf_attn.k.weight.data,
            n_heads=local.n_heads,
            head_size=local.key_value_proj_dim,
        )
    )
    local.v_weight.data.copy_(
        linear_to_qkv_weight(
            hf_attn.v.weight.data,
            n_heads=local.n_heads,
            head_size=local.key_value_proj_dim,
        )
    )
    local.o_weight.data.copy_(
        linear_to_output_weight(
            hf_attn.o.weight.data,
            n_heads=local.n_heads,
            head_size=local.key_value_proj_dim,
        )
    )


def _load_ff(local: T5LayerFF, hf_ff: HfLayerFF) -> None:
    local.layer_norm.weight.data.copy_(hf_ff.layer_norm.weight.data)
    copy_linear(local.DenseReluDense.wi_0, hf_ff.DenseReluDense.wi_0)
    copy_linear(local.DenseReluDense.wi_1, hf_ff.DenseReluDense.wi_1)
    copy_linear(local.DenseReluDense.wo, hf_ff.DenseReluDense.wo)


def _load_encoder_block(local: T5EncoderBlock, hf_block: HfBlock) -> None:
    local.layer_norm.weight.data.copy_(
        hf_block.layer[0].layer_norm.weight.data
    )
    _load_attn(local.self_attn, hf_block.layer[0].SelfAttention)
    _load_ff(local.ff, hf_block.layer[1])


def _load_decoder_block(local: T5DecoderBlock, hf_block: HfBlock) -> None:
    local.layer_norm_0.weight.data.copy_(
        hf_block.layer[0].layer_norm.weight.data
    )
    _load_attn(local.self_attn, hf_block.layer[0].SelfAttention)
    local.layer_norm_1.weight.data.copy_(
        hf_block.layer[1].layer_norm.weight.data
    )
    _load_attn(local.cross_attn, hf_block.layer[1].EncDecAttention)
    _load_ff(local.ff, hf_block.layer[2])


# ---------------------------------------------------------------------------
# FF (was t5_ff.cc)
# ---------------------------------------------------------------------------


def test_t5_ff_forward_backward_matches_hf():
    torch.manual_seed(1)
    hf_cfg = _hf_cfg()
    hf_ff = HfLayerFF(hf_cfg).eval().float()
    local = T5LayerFF(t5_config_from_hf(hf_cfg)).eval().float()
    _load_ff(local, hf_ff)
    local_n = local.to("nntile")

    x = torch.randn(2, 8, hf_cfg.d_model, requires_grad=True)
    y_ref = hf_ff(x)
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)

    x_n = contiguous_to_nntile(x.detach()).requires_grad_(True)
    y = local_n(x_n)
    assert_close(y, y_ref.detach(), rtol=RTOL, atol=ATOL)
    gx, gw = torch.autograd.grad(
        y,
        (x_n, local_n.DenseReluDense.wi_0.weight),
        contiguous_to_nntile(grad),
    )
    assert_close(gx, x.grad, rtol=1e-3, atol=BWD_ATOL)
    assert_close(
        gw, hf_ff.DenseReluDense.wi_0.weight.grad, rtol=1e-3, atol=BWD_ATOL
    )


# ---------------------------------------------------------------------------
# Attention: self no-mask, self causal, cross same seq
# (was t5_attention.cc)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True])
def test_t5_self_attention_forward_backward_matches_hf(causal):
    torch.manual_seed(10 + int(causal))
    hf_cfg = _hf_cfg()
    hf_attn = (
        HfAttention(
            hf_cfg,
            has_relative_attention_bias=False,
            layer_idx=0,
        )
        .eval()
        .float()
    )
    local = T5Attention(t5_config_from_hf(hf_cfg), is_decoder=causal).eval()
    local.float()
    _load_attn(local, hf_attn)
    local_n = local.to("nntile")

    b, s, h = 2, 8, hf_cfg.d_model
    x = torch.randn(b, s, h, requires_grad=True)
    mask = additive_causal_mask(b, s) if causal else None
    y_ref = hf_attn(
        x,
        mask=mask,
        cache_position=torch.arange(s),
    )[0]
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)

    x_n = contiguous_to_nntile(x.detach()).requires_grad_(True)
    y = local_n(
        x_n,
        attn_mask=None,
        is_causal=causal,
    )
    assert_close(y, y_ref.detach(), rtol=RTOL, atol=ATTN_ATOL)
    (gx,) = torch.autograd.grad(y, x_n, contiguous_to_nntile(grad))
    assert_close(gx, x.grad, rtol=1e-3, atol=BWD_ATOL)


def test_t5_cross_attention_forward_backward_matches_hf():
    torch.manual_seed(12)
    hf_cfg = _hf_cfg()
    hf_attn = (
        HfAttention(
            _decoder_hf_cfg(hf_cfg),
            has_relative_attention_bias=False,
            layer_idx=0,
        )
        .eval()
        .float()
    )
    local = (
        T5Attention(
            t5_config_from_hf(hf_cfg),
            is_decoder=True,
        )
        .eval()
        .float()
    )
    _load_attn(local, hf_attn)
    local_n = local.to("nntile")

    b, s, h = 2, 8, hf_cfg.d_model
    dec = torch.randn(b, s, h, requires_grad=True)
    enc = torch.randn(b, s, h, requires_grad=True)
    y_ref = hf_attn(
        dec,
        key_value_states=enc,
        cache_position=torch.arange(s),
    )[0]
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)

    dec_n = contiguous_to_nntile(dec.detach()).requires_grad_(True)
    enc_n = contiguous_to_nntile(enc.detach()).requires_grad_(True)
    y = local_n(
        dec_n,
        key_value_states=enc_n,
        is_causal=False,
    )
    assert_close(y, y_ref.detach(), rtol=RTOL, atol=ATTN_ATOL)
    gx, ge = torch.autograd.grad(
        y,
        (dec_n, enc_n),
        contiguous_to_nntile(grad),
    )
    assert_close(gx, dec.grad, rtol=1e-3, atol=BWD_ATOL)
    assert_close(ge, enc.grad, rtol=1e-3, atol=BWD_ATOL)


# ---------------------------------------------------------------------------
# Encoder / decoder blocks (was t5_encoder.cc / t5_decoder.cc)
# ---------------------------------------------------------------------------


def test_t5_encoder_block_forward_backward_matches_hf():
    torch.manual_seed(20)
    hf_cfg = _hf_cfg()
    hf_block = (
        HfBlock(
            hf_cfg,
            has_relative_attention_bias=False,
            layer_idx=0,
        )
        .eval()
        .float()
    )
    local = T5EncoderBlock(t5_config_from_hf(hf_cfg)).eval().float()
    _load_encoder_block(local, hf_block)
    local_n = local.to("nntile")

    b, s, h = 2, 8, hf_cfg.d_model
    x = torch.randn(b, s, h, requires_grad=True)
    y_ref = hf_block(x, cache_position=torch.arange(s))[0]
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)

    x_n = contiguous_to_nntile(x.detach()).requires_grad_(True)
    y = local_n(x_n)
    assert_close(y, y_ref.detach(), rtol=RTOL, atol=ATTN_ATOL)
    (gx,) = torch.autograd.grad(y, x_n, contiguous_to_nntile(grad))
    assert_close(gx, x.grad, rtol=1e-3, atol=BWD_ATOL)


def test_t5_decoder_block_forward_matches_hf():
    torch.manual_seed(21)
    hf_cfg = _hf_cfg()
    hf_block = (
        HfBlock(
            _decoder_hf_cfg(hf_cfg),
            has_relative_attention_bias=False,
            layer_idx=0,
        )
        .eval()
        .float()
    )
    local = T5DecoderBlock(t5_config_from_hf(hf_cfg)).eval().float()
    _load_decoder_block(local, hf_block)
    local_n = local.to("nntile")

    b, s, h = 2, 8, hf_cfg.d_model
    dec = torch.randn(b, s, h)
    enc = torch.randn(b, s, h)
    self_mask = additive_causal_mask(b, s)
    with torch.no_grad():
        ref = hf_block(
            dec,
            attention_mask=self_mask,
            encoder_hidden_states=enc,
            cache_position=torch.arange(s),
        )[0]
        out = local_n(
            contiguous_to_nntile(dec),
            contiguous_to_nntile(enc),
            self_attn_mask=None,
        )
    assert_close(out, ref, rtol=RTOL, atol=ATTN_ATOL)


# ---------------------------------------------------------------------------
# Model + Conditional generation (was t5_model.cc / t5_causal.cc)
# ---------------------------------------------------------------------------


def test_t5_model_encoder_decoder_hidden_forward_matches_hf():
    hf_cfg = _hf_cfg()
    hf, local = _make_models(hf_cfg)
    enc_ids = torch.randint(0, hf_cfg.vocab_size, (2, 8))
    dec_ids = torch.randint(0, hf_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        enc_ref = hf.encoder(input_ids=enc_ids).last_hidden_state
        ref = hf.decoder(
            input_ids=dec_ids,
            encoder_hidden_states=enc_ref,
        ).last_hidden_state
        out = local.model(
            contiguous_to_nntile(enc_ids),
            contiguous_to_nntile(dec_ids),
        )
    assert_close(out, ref, rtol=RTOL, atol=ATTN_ATOL)


def test_t5_conditional_generation_forward_matches_hf():
    hf_cfg = _hf_cfg()
    hf, local = _make_models(hf_cfg)
    enc_ids = torch.randint(0, hf_cfg.vocab_size, (2, 8))
    dec_ids = torch.randint(0, hf_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        ref = hf(input_ids=enc_ids, decoder_input_ids=dec_ids).logits
        out = local(
            contiguous_to_nntile(enc_ids),
            contiguous_to_nntile(dec_ids),
        )
    assert_close(out, ref, rtol=RTOL, atol=ATTN_ATOL)


def test_t5_conditional_generation_backward_matches_hf():
    hf_cfg = _hf_cfg()
    # Local T5 stays untied. HF 5 T5Config always reports tied embeddings
    # (``tie_word_embeddings`` is not an __init__ argument).
    hf, local = _make_models(hf_cfg)
    for p in hf.parameters():
        p.requires_grad_(True)
    for p in local.parameters():
        p.requires_grad_(True)

    enc_ids = torch.randint(0, hf_cfg.vocab_size, (2, 8))
    dec_ids = torch.randint(0, hf_cfg.vocab_size, (2, 8))
    grad = torch.randn(2, 8, hf_cfg.vocab_size)
    hf(input_ids=enc_ids, decoder_input_ids=dec_ids).logits.backward(grad)
    logits = local(
        contiguous_to_nntile(enc_ids),
        contiguous_to_nntile(dec_ids),
    )
    gw_q, gw_lm, gw_shared = torch.autograd.grad(
        logits,
        (
            local.model.encoder.block[0].self_attn.q_weight,
            local.lm_head.weight,
            local.model.shared.weight,
        ),
        contiguous_to_nntile(grad),
    )
    assert_close(
        qkv_to_linear_weight(nntile_cpu(gw_q)),
        hf.encoder.block[0].layer[0].SelfAttention.q.weight.grad,
        rtol=1e-3,
        atol=BWD_ATOL,
    )
    if hf.config.tie_word_embeddings:
        # HF 5 T5 ties ``lm_head`` to ``shared``; local copies stay untied.
        return
    assert_close(gw_lm, hf.lm_head.weight.grad, rtol=1e-3, atol=BWD_ATOL)
    assert_close(gw_shared, hf.shared.weight.grad, rtol=1e-3, atol=BWD_ATOL)
