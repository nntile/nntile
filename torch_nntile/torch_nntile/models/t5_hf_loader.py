# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/t5_hf_loader.py
# Convert weights between HuggingFace T5 and torch_nntile T5.

"""Bidirectional HF <-> NNTile weight conversion for T5.

Local ``T5Attention`` has no relative-position bias. Loaders copy Q/K/V/O and
FF weights; callers should disable HF relative bias before comparing forwards
(see :func:`disable_t5_relative_attention_bias`).
"""

from __future__ import annotations

import torch
from transformers import T5Config as HfT5Config
from transformers import (
    T5ForConditionalGeneration as HfT5ForConditionalGeneration,
)

from torch_nntile.models.hf_rope_layout import copy_linear
from torch_nntile.nn.linear import (
    linear_to_output_weight,
    linear_to_qkv_weight,
    output_to_linear_weight,
    qkv_to_linear_weight,
)
from torch_nntile.models.t5 import T5Config, T5ForConditionalGeneration


def t5_config_from_hf(hf: HfT5Config) -> T5Config:
    """Build a local ``T5Config`` from an HF config."""
    decoder_start = getattr(hf, "decoder_start_token_id", None)
    if decoder_start is None:
        decoder_start = hf.pad_token_id
    return T5Config(
        vocab_size=int(hf.vocab_size),
        d_model=int(hf.d_model),
        d_kv=int(hf.d_kv),
        d_ff=int(hf.d_ff),
        num_layers=int(hf.num_layers),
        num_decoder_layers=int(
            getattr(hf, "num_decoder_layers", hf.num_layers)
        ),
        num_heads=int(hf.num_heads),
        relative_attention_num_buckets=int(
            hf.relative_attention_num_buckets
        ),
        layer_norm_epsilon=float(hf.layer_norm_epsilon),
        dropout_rate=float(hf.dropout_rate),
        feed_forward_proj=str(hf.feed_forward_proj),
        is_gated_act=bool(getattr(hf, "is_gated_act", True)),
        tie_word_embeddings=False,  # local models stay untied (migration debt)
        pad_token_id=int(hf.pad_token_id),
        eos_token_id=int(hf.eos_token_id),
        decoder_start_token_id=int(decoder_start),
    )


def disable_t5_relative_attention_bias(
    hf: HfT5ForConditionalGeneration,
) -> None:
    """Turn off relative bias on every encoder/decoder self-attention."""
    for block in hf.encoder.block:
        block.layer[0].SelfAttention.has_relative_attention_bias = False
    for block in hf.decoder.block:
        block.layer[0].SelfAttention.has_relative_attention_bias = False


def _load_attn(dst, src) -> None:
    n_heads = dst.n_heads
    head_size = dst.key_value_proj_dim
    dst.q_weight.data.copy_(
        linear_to_qkv_weight(
            src.q.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    dst.k_weight.data.copy_(
        linear_to_qkv_weight(
            src.k.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    dst.v_weight.data.copy_(
        linear_to_qkv_weight(
            src.v.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    dst.o_weight.data.copy_(
        linear_to_output_weight(
            src.o.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )


def _export_attn(dst, src) -> None:
    dst.q.weight.data.copy_(qkv_to_linear_weight(src.q_weight.data))
    dst.k.weight.data.copy_(qkv_to_linear_weight(src.k_weight.data))
    dst.v.weight.data.copy_(qkv_to_linear_weight(src.v_weight.data))
    dst.o.weight.data.copy_(output_to_linear_weight(src.o_weight.data))


def _load_ff(dst_ff, src_ff) -> None:
    dst_ff.layer_norm.weight.data.copy_(src_ff.layer_norm.weight.data)
    dense = src_ff.DenseReluDense
    copy_linear(dst_ff.DenseReluDense.wi_0, dense.wi_0)
    copy_linear(dst_ff.DenseReluDense.wi_1, dense.wi_1)
    copy_linear(dst_ff.DenseReluDense.wo, dense.wo)


def _load_encoder_block(dst, src) -> None:
    dst.layer_norm.weight.data.copy_(src.layer[0].layer_norm.weight.data)
    _load_attn(dst.self_attn, src.layer[0].SelfAttention)
    _load_ff(dst.ff, src.layer[1])


def _load_decoder_block(dst, src) -> None:
    dst.layer_norm_0.weight.data.copy_(src.layer[0].layer_norm.weight.data)
    _load_attn(dst.self_attn, src.layer[0].SelfAttention)
    dst.layer_norm_1.weight.data.copy_(src.layer[1].layer_norm.weight.data)
    _load_attn(dst.cross_attn, src.layer[1].EncDecAttention)
    _load_ff(dst.ff, src.layer[2])


def load_hf_into_t5(
    minimal: T5ForConditionalGeneration,
    hf: HfT5ForConditionalGeneration,
) -> None:
    """Copy HF weights into ``minimal`` (CPU; call ``.to('nntile')`` after)."""
    dst = minimal.model

    dst.shared.weight.data.copy_(hf.shared.weight.data)
    for dst_b, src_b in zip(dst.encoder.block, hf.encoder.block):
        _load_encoder_block(dst_b, src_b)
    dst.encoder.final_layer_norm.weight.data.copy_(
        hf.encoder.final_layer_norm.weight.data
    )
    for dst_b, src_b in zip(dst.decoder.block, hf.decoder.block):
        _load_decoder_block(dst_b, src_b)
    dst.decoder.final_layer_norm.weight.data.copy_(
        hf.decoder.final_layer_norm.weight.data
    )

    # Always keep an independent lm_head (tying deferred; migration debt).
    minimal.lm_head.weight.data.copy_(hf.lm_head.weight.data)


def export_t5_to_hf_state_dict(
    minimal: T5ForConditionalGeneration,
    *,
    config: HfT5Config | None = None,
) -> dict[str, torch.Tensor]:
    """Export local T5 CPU weights as an HF state_dict."""
    cfg = minimal.config
    if config is None:
        config = HfT5Config(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            d_kv=cfg.d_kv,
            d_ff=cfg.d_ff,
            num_layers=cfg.num_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            num_heads=cfg.num_heads,
            relative_attention_num_buckets=(
                cfg.relative_attention_num_buckets
            ),
            layer_norm_epsilon=cfg.layer_norm_epsilon,
            dropout_rate=cfg.dropout_rate,
            feed_forward_proj=cfg.feed_forward_proj,
            is_gated_act=cfg.is_gated_act,
            tie_word_embeddings=cfg.tie_word_embeddings,
            pad_token_id=cfg.pad_token_id,
            eos_token_id=cfg.eos_token_id,
            decoder_start_token_id=cfg.decoder_start_token_id,
        )
    config._attn_implementation = "eager"
    hf = HfT5ForConditionalGeneration(config).float()
    src = minimal.model

    hf.shared.weight.data.copy_(src.shared.weight.data)
    for dst_b, src_b in zip(hf.encoder.block, src.encoder.block):
        dst_b.layer[0].layer_norm.weight.data.copy_(
            src_b.layer_norm.weight.data
        )
        _export_attn(dst_b.layer[0].SelfAttention, src_b.self_attn)
        _load_ff(dst_b.layer[1], src_b.ff)
    hf.encoder.final_layer_norm.weight.data.copy_(
        src.encoder.final_layer_norm.weight.data
    )
    for dst_b, src_b in zip(hf.decoder.block, src.decoder.block):
        dst_b.layer[0].layer_norm.weight.data.copy_(
            src_b.layer_norm_0.weight.data
        )
        _export_attn(dst_b.layer[0].SelfAttention, src_b.self_attn)
        dst_b.layer[1].layer_norm.weight.data.copy_(
            src_b.layer_norm_1.weight.data
        )
        _export_attn(dst_b.layer[1].EncDecAttention, src_b.cross_attn)
        _load_ff(dst_b.layer[2], src_b.ff)
    hf.decoder.final_layer_norm.weight.data.copy_(
        src.decoder.final_layer_norm.weight.data
    )

    # Always export an independent lm_head; HF may re-tie for its layout.
    hf.lm_head.weight.data.copy_(minimal.lm_head.weight.data)
    # Keep lm_head untied (migration debt).

    with torch.no_grad():
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in hf.state_dict().items()
        }


__all__ = [
    "disable_t5_relative_attention_bias",
    "export_t5_to_hf_state_dict",
    "load_hf_into_t5",
    "t5_config_from_hf",
]
