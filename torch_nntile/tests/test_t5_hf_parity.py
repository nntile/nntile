# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_t5_hf_parity.py
# T5 layer + full-model parity vs HuggingFace T5ForConditionalGeneration.

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("transformers")

import torch
from transformers import T5Config as HfT5Config
from transformers import T5ForConditionalGeneration as HfT5
from transformers.models.t5.modeling_t5 import (
    T5Attention as HfAttention,
    T5Block as HfBlock,
    T5LayerFF as HfLayerFF,
)

import torch_nntile
from torch_nntile import _C
from torch_nntile.models.hf_rope_layout import copy_linear
from torch_nntile.models.t5 import (
    T5Attention,
    T5Config,
    T5DecoderBlock,
    T5DenseGatedActDense,
    T5EncoderBlock,
    T5ForConditionalGeneration,
    T5LayerFF,
)
from torch_nntile.models.t5_hf_loader import (
    disable_t5_relative_attention_bias,
    load_hf_into_t5,
    t5_config_from_hf,
)
from parity_helpers import assert_close, contiguous_to_nntile


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)

RTOL = 1e-4
ATOL = 1e-4


@pytest.fixture
def tiny_hf_config() -> HfT5Config:
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


def _make_models(hf_cfg: HfT5Config):
    torch.manual_seed(0)
    hf = HfT5(hf_cfg).eval().float()
    disable_t5_relative_attention_bias(hf)
    local_cfg = t5_config_from_hf(hf_cfg)
    # Keep untied for nntile scalar-scale path simplicity in tiny tests.
    local_cfg.tie_word_embeddings = False
    hf_cfg.tie_word_embeddings = False
    minimal = T5ForConditionalGeneration(local_cfg).eval().float()
    load_hf_into_t5(minimal, hf)
    minimal = minimal.to("nntile")
    return hf, minimal


def _load_attn(local: T5Attention, hf_attn: HfAttention) -> None:
    copy_linear(local.q, hf_attn.q)
    copy_linear(local.k, hf_attn.k)
    copy_linear(local.v, hf_attn.v)
    copy_linear(local.o, hf_attn.o)


def test_t5_ff_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(1)
    hf_ff = HfLayerFF(tiny_hf_config).eval().float()
    local = T5LayerFF(t5_config_from_hf(tiny_hf_config)).eval().float()
    local.layer_norm.weight.data.copy_(hf_ff.layer_norm.weight.data)
    copy_linear(local.DenseReluDense.wi_0, hf_ff.DenseReluDense.wi_0)
    copy_linear(local.DenseReluDense.wi_1, hf_ff.DenseReluDense.wi_1)
    copy_linear(local.DenseReluDense.wo, hf_ff.DenseReluDense.wo)
    local = local.to("nntile")
    x = torch.randn(2, 8, tiny_hf_config.d_model)
    with torch.no_grad():
        ref = hf_ff(x)
        out = local(contiguous_to_nntile(x))
    assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_t5_attention_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(2)
    hf_attn = HfAttention(
        tiny_hf_config, has_relative_attention_bias=False
    ).eval().float()
    local = T5Attention(
        t5_config_from_hf(tiny_hf_config), is_decoder=False
    ).eval().float()
    _load_attn(local, hf_attn)
    local = local.to("nntile")
    x = torch.randn(2, 8, tiny_hf_config.d_model)
    cache_position = torch.arange(8)
    with torch.no_grad():
        ref = hf_attn(x, cache_position=cache_position)[0]
        out = local(contiguous_to_nntile(x), is_causal=False)
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


def test_t5_encoder_block_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(3)
    hf_block = HfBlock(tiny_hf_config, has_relative_attention_bias=False).eval().float()
    local = T5EncoderBlock(t5_config_from_hf(tiny_hf_config)).eval().float()
    local.layer_norm.weight.data.copy_(
        hf_block.layer[0].layer_norm.weight.data
    )
    _load_attn(local.self_attn, hf_block.layer[0].SelfAttention)
    local.ff.layer_norm.weight.data.copy_(
        hf_block.layer[1].layer_norm.weight.data
    )
    copy_linear(
        local.ff.DenseReluDense.wi_0, hf_block.layer[1].DenseReluDense.wi_0
    )
    copy_linear(
        local.ff.DenseReluDense.wi_1, hf_block.layer[1].DenseReluDense.wi_1
    )
    copy_linear(
        local.ff.DenseReluDense.wo, hf_block.layer[1].DenseReluDense.wo
    )
    local = local.to("nntile")
    x = torch.randn(2, 8, tiny_hf_config.d_model)
    cache_position = torch.arange(8)
    with torch.no_grad():
        ref = hf_block(x, cache_position=cache_position)[0]
        out = local(contiguous_to_nntile(x))
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


def test_t5_cross_attention_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(4)
    hf = HfT5(tiny_hf_config).eval().float()
    disable_t5_relative_attention_bias(hf)
    hf_cross = hf.decoder.block[0].layer[1].EncDecAttention
    local = T5Attention(
        t5_config_from_hf(tiny_hf_config), is_decoder=True
    ).eval().float()
    _load_attn(local, hf_cross)
    local = local.to("nntile")
    dec = torch.randn(2, 8, tiny_hf_config.d_model)
    enc = torch.randn(2, 8, tiny_hf_config.d_model)
    cache_position = torch.arange(8)
    with torch.no_grad():
        ref = hf_cross(
            dec,
            key_value_states=enc,
            cache_position=cache_position,
        )[0]
        out = local(
            contiguous_to_nntile(dec),
            key_value_states=contiguous_to_nntile(enc),
            is_causal=False,
        )
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


def test_t5_conditional_forward_matches_hf(tiny_hf_config):
    hf, minimal = _make_models(tiny_hf_config)
    enc_ids = torch.randint(0, tiny_hf_config.vocab_size, (2, 8))
    dec_ids = torch.randint(0, tiny_hf_config.vocab_size, (2, 8))
    with torch.no_grad():
        ref = hf(input_ids=enc_ids, decoder_input_ids=dec_ids).logits
        out = minimal(
            contiguous_to_nntile(enc_ids),
            contiguous_to_nntile(dec_ids),
        )
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


def test_t5_conditional_backward_matches_hf(tiny_hf_config):
    hf, minimal = _make_models(tiny_hf_config)
    for p in hf.parameters():
        p.requires_grad_(True)
    for p in minimal.parameters():
        p.requires_grad_(True)
    enc_ids = torch.randint(0, tiny_hf_config.vocab_size, (2, 8))
    dec_ids = torch.randint(0, tiny_hf_config.vocab_size, (2, 8))
    grad = torch.randn(2, 8, tiny_hf_config.vocab_size)
    logits_ref = hf(input_ids=enc_ids, decoder_input_ids=dec_ids).logits
    logits_ref.backward(grad)
    logits = minimal(
        contiguous_to_nntile(enc_ids),
        contiguous_to_nntile(dec_ids),
    )
    (gw,) = torch.autograd.grad(
        logits,
        minimal.model.encoder.block[0].self_attn.q.weight,
        grad_outputs=contiguous_to_nntile(grad),
    )
    assert_close(
        gw,
        hf.encoder.block[0].layer[0].SelfAttention.q.weight.grad,
        rtol=1e-3,
        atol=1e-3,
    )
