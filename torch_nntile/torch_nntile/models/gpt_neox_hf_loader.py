# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/gpt_neox_hf_loader.py
# Convert weights between HuggingFace GPT-NeoX and torch_nntile GPTNeoXCausal.

"""Bidirectional HF ↔ NNTile weight conversion for GPT-NeoX."""

from __future__ import annotations

import torch
from transformers import GPTNeoXConfig as HfGPTNeoXConfig
from transformers import GPTNeoXForCausalLM

from torch_nntile.models.gpt_neox import GPTNeoXCausal, GPTNeoXConfig
from torch_nntile.models.hf_rope_layout import (
    copy_linear,
    hf_to_nntile_fused_qkv_bias,
    hf_to_nntile_fused_qkv_weight,
    nntile_to_hf_fused_qkv_bias,
    nntile_to_hf_fused_qkv_weight,
)


def gpt_neox_config_from_hf(hf: HfGPTNeoXConfig) -> GPTNeoXConfig:
    """Build a local ``GPTNeoXConfig`` from an HF config."""
    return GPTNeoXConfig(
        vocab_size=int(hf.vocab_size),
        hidden_size=int(hf.hidden_size),
        intermediate_size=int(hf.intermediate_size),
        num_hidden_layers=int(hf.num_hidden_layers),
        num_attention_heads=int(hf.num_attention_heads),
        max_position_embeddings=int(hf.max_position_embeddings),
        layer_norm_eps=float(hf.layer_norm_eps),
        rotary_pct=float(hf.rotary_pct),
        rotary_emb_base=float(hf.rotary_emb_base),
        use_parallel_residual=bool(hf.use_parallel_residual),
        attention_bias=bool(getattr(hf, "attention_bias", True)),
        tie_word_embeddings=False,  # local models stay untied (migration debt)
    )


def _load_attn_from_hf(dst_attn, src_attn, cfg: GPTNeoXConfig) -> None:
    n_heads = cfg.num_attention_heads
    head_dim = cfg.head_dim
    pct = cfg.rotary_pct
    dst_attn.query_key_value.weight.data.copy_(
        hf_to_nntile_fused_qkv_weight(
            src_attn.query_key_value.weight.data,
            n_heads=n_heads,
            head_dim=head_dim,
            rotary_pct=pct,
        )
    )
    copy_linear(dst_attn.dense, src_attn.dense)
    if (
        dst_attn.query_key_value.bias is not None
        and src_attn.query_key_value.bias is not None
    ):
        dst_attn.query_key_value.bias.data.copy_(
            hf_to_nntile_fused_qkv_bias(
                src_attn.query_key_value.bias.data,
                n_heads=n_heads,
                head_dim=head_dim,
                rotary_pct=pct,
            )
        )


def _export_attn_to_hf(src_attn, dst_attn, cfg: GPTNeoXConfig) -> None:
    n_heads = cfg.num_attention_heads
    head_dim = cfg.head_dim
    pct = cfg.rotary_pct
    dst_attn.query_key_value.weight.data.copy_(
        nntile_to_hf_fused_qkv_weight(
            src_attn.query_key_value.weight.data,
            n_heads=n_heads,
            head_dim=head_dim,
            rotary_pct=pct,
        )
    )
    copy_linear(dst_attn.dense, src_attn.dense)
    if (
        src_attn.query_key_value.bias is not None
        and dst_attn.query_key_value.bias is not None
    ):
        dst_attn.query_key_value.bias.data.copy_(
            nntile_to_hf_fused_qkv_bias(
                src_attn.query_key_value.bias.data,
                n_heads=n_heads,
                head_dim=head_dim,
                rotary_pct=pct,
            )
        )


def load_hf_into_gpt_neox_causal(
    minimal: GPTNeoXCausal,
    hf: GPTNeoXForCausalLM,
) -> None:
    """Copy HF weights into ``minimal`` (CPU; call ``.to('nntile')`` after)."""
    cfg = minimal.config
    src = hf.gpt_neox
    dst = minimal.gpt_neox

    dst.embed_in.weight.data.copy_(src.embed_in.weight.data)
    dst.final_layer_norm.weight.data.copy_(src.final_layer_norm.weight.data)
    dst.final_layer_norm.bias.data.copy_(src.final_layer_norm.bias.data)

    for dst_layer, src_layer in zip(dst.layers, src.layers):
        dst_layer.input_layernorm.weight.data.copy_(
            src_layer.input_layernorm.weight.data
        )
        dst_layer.input_layernorm.bias.data.copy_(
            src_layer.input_layernorm.bias.data
        )
        dst_layer.post_attention_layernorm.weight.data.copy_(
            src_layer.post_attention_layernorm.weight.data
        )
        dst_layer.post_attention_layernorm.bias.data.copy_(
            src_layer.post_attention_layernorm.bias.data
        )
        _load_attn_from_hf(dst_layer.attention, src_layer.attention, cfg)
        copy_linear(dst_layer.mlp.dense_h_to_4h, src_layer.mlp.dense_h_to_4h)
        copy_linear(dst_layer.mlp.dense_4h_to_h, src_layer.mlp.dense_4h_to_h)

    # Always keep an independent embed_out (tying deferred; migration debt).
    minimal.embed_out.weight.data.copy_(hf.embed_out.weight.data)


def export_gpt_neox_causal_to_hf_state_dict(
    minimal: GPTNeoXCausal,
    *,
    config: HfGPTNeoXConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Export local GPT-NeoX CPU weights as an HF state_dict."""
    cfg = minimal.config
    if config is None:
        config = HfGPTNeoXConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            max_position_embeddings=cfg.max_position_embeddings,
            layer_norm_eps=cfg.layer_norm_eps,
            rotary_pct=cfg.rotary_pct,
            rotary_emb_base=cfg.rotary_emb_base,
            use_parallel_residual=cfg.use_parallel_residual,
            attention_bias=cfg.attention_bias,
            tie_word_embeddings=cfg.tie_word_embeddings,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
    config._attn_implementation = "eager"
    hf = GPTNeoXForCausalLM(config).float()
    src = minimal.gpt_neox
    dst = hf.gpt_neox

    dst.embed_in.weight.data.copy_(src.embed_in.weight.data)
    dst.final_layer_norm.weight.data.copy_(src.final_layer_norm.weight.data)
    dst.final_layer_norm.bias.data.copy_(src.final_layer_norm.bias.data)

    for dst_layer, src_layer in zip(dst.layers, src.layers):
        dst_layer.input_layernorm.weight.data.copy_(
            src_layer.input_layernorm.weight.data
        )
        dst_layer.input_layernorm.bias.data.copy_(
            src_layer.input_layernorm.bias.data
        )
        dst_layer.post_attention_layernorm.weight.data.copy_(
            src_layer.post_attention_layernorm.weight.data
        )
        dst_layer.post_attention_layernorm.bias.data.copy_(
            src_layer.post_attention_layernorm.bias.data
        )
        _export_attn_to_hf(src_layer.attention, dst_layer.attention, cfg)
        copy_linear(dst_layer.mlp.dense_h_to_4h, src_layer.mlp.dense_h_to_4h)
        copy_linear(dst_layer.mlp.dense_4h_to_h, src_layer.mlp.dense_4h_to_h)

    hf.embed_out.weight.data.copy_(minimal.embed_out.weight.data)
    # Keep embed_out untied (migration debt).

    with torch.no_grad():
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in hf.state_dict().items()
        }


__all__ = [
    "export_gpt_neox_causal_to_hf_state_dict",
    "gpt_neox_config_from_hf",
    "load_hf_into_gpt_neox_causal",
]
