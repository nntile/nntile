# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/llama_hf_loader.py
# Convert weights between HuggingFace Llama and torch_nntile LlamaCausal.

"""Bidirectional HF ↔ NNTile weight conversion for Llama."""

from __future__ import annotations

import torch
from transformers import LlamaConfig as HfLlamaConfig
from transformers import LlamaForCausalLM

from torch_nntile.models.hf_rope_layout import (
    copy_linear,
    hf_to_nntile_qkv_bias,
    hf_to_nntile_qkv_weight,
    nntile_to_hf_qkv_bias,
    nntile_to_hf_qkv_weight,
)
from torch_nntile.models.llama import LlamaCausal, LlamaConfig


def llama_config_from_hf(hf: HfLlamaConfig) -> LlamaConfig:
    """Build a local ``LlamaConfig`` from an HF config."""
    n_kv = getattr(hf, "num_key_value_heads", None)
    if n_kv is None:
        n_kv = hf.num_attention_heads
    return LlamaConfig(
        vocab_size=int(hf.vocab_size),
        hidden_size=int(hf.hidden_size),
        intermediate_size=int(hf.intermediate_size),
        num_hidden_layers=int(hf.num_hidden_layers),
        num_attention_heads=int(hf.num_attention_heads),
        num_key_value_heads=int(n_kv),
        max_position_embeddings=int(hf.max_position_embeddings),
        rms_norm_eps=float(hf.rms_norm_eps),
        rope_theta=float(getattr(hf, "rope_theta", 10000.0)),
        attention_bias=bool(getattr(hf, "attention_bias", False)),
        mlp_bias=bool(getattr(hf, "mlp_bias", False)),
        tie_word_embeddings=False,  # local models stay untied (migration debt)
    )


def _load_attn_from_hf(dst_attn, src_attn, cfg: LlamaConfig) -> None:
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    dst_attn.q_proj.weight.data.copy_(
        hf_to_nntile_qkv_weight(
            src_attn.q_proj.weight.data,
            n_heads=n_heads,
            head_dim=head_dim,
        )
    )
    dst_attn.k_proj.weight.data.copy_(
        hf_to_nntile_qkv_weight(
            src_attn.k_proj.weight.data,
            n_heads=n_kv,
            head_dim=head_dim,
        )
    )
    copy_linear(dst_attn.v_proj, src_attn.v_proj)
    copy_linear(dst_attn.o_proj, src_attn.o_proj)
    if dst_attn.q_proj.bias is not None and src_attn.q_proj.bias is not None:
        dst_attn.q_proj.bias.data.copy_(
            hf_to_nntile_qkv_bias(
                src_attn.q_proj.bias.data,
                n_heads=n_heads,
                head_dim=head_dim,
            )
        )
    if dst_attn.k_proj.bias is not None and src_attn.k_proj.bias is not None:
        dst_attn.k_proj.bias.data.copy_(
            hf_to_nntile_qkv_bias(
                src_attn.k_proj.bias.data,
                n_heads=n_kv,
                head_dim=head_dim,
            )
        )


def _export_attn_to_hf(src_attn, dst_attn, cfg: LlamaConfig) -> None:
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    dst_attn.q_proj.weight.data.copy_(
        nntile_to_hf_qkv_weight(
            src_attn.q_proj.weight.data,
            n_heads=n_heads,
            head_dim=head_dim,
        )
    )
    dst_attn.k_proj.weight.data.copy_(
        nntile_to_hf_qkv_weight(
            src_attn.k_proj.weight.data,
            n_heads=n_kv,
            head_dim=head_dim,
        )
    )
    copy_linear(dst_attn.v_proj, src_attn.v_proj)
    copy_linear(dst_attn.o_proj, src_attn.o_proj)
    if src_attn.q_proj.bias is not None and dst_attn.q_proj.bias is not None:
        dst_attn.q_proj.bias.data.copy_(
            nntile_to_hf_qkv_bias(
                src_attn.q_proj.bias.data,
                n_heads=n_heads,
                head_dim=head_dim,
            )
        )
    if src_attn.k_proj.bias is not None and dst_attn.k_proj.bias is not None:
        dst_attn.k_proj.bias.data.copy_(
            nntile_to_hf_qkv_bias(
                src_attn.k_proj.bias.data,
                n_heads=n_kv,
                head_dim=head_dim,
            )
        )


def load_hf_into_llama_causal(
    minimal: LlamaCausal,
    hf: LlamaForCausalLM,
) -> None:
    """Copy HF weights into ``minimal`` (CPU; call ``.to('nntile')`` after)."""
    cfg = minimal.config
    src = hf.model
    dst = minimal.model

    dst.embed_tokens.weight.data.copy_(src.embed_tokens.weight.data)
    dst.norm.weight.data.copy_(src.norm.weight.data)

    for dst_layer, src_layer in zip(dst.layers, src.layers):
        dst_layer.input_layernorm.weight.data.copy_(
            src_layer.input_layernorm.weight.data
        )
        dst_layer.post_attention_layernorm.weight.data.copy_(
            src_layer.post_attention_layernorm.weight.data
        )
        _load_attn_from_hf(dst_layer.self_attn, src_layer.self_attn, cfg)
        copy_linear(dst_layer.mlp.gate_proj, src_layer.mlp.gate_proj)
        copy_linear(dst_layer.mlp.up_proj, src_layer.mlp.up_proj)
        copy_linear(dst_layer.mlp.down_proj, src_layer.mlp.down_proj)

    if cfg.tie_word_embeddings:
        # Untied locally: copy shared embedding values into lm_head by value.
        minimal.lm_head.weight.data.copy_(
            minimal.model.embed_tokens.weight.data
        )
    else:
        minimal.lm_head.weight.data.copy_(hf.lm_head.weight.data)


def export_llama_causal_to_hf_state_dict(
    minimal: LlamaCausal,
    *,
    config: HfLlamaConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Export local Llama CPU weights as an HF ``LlamaForCausalLM`` state_dict."""
    cfg = minimal.config
    if config is None:
        config = HfLlamaConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            max_position_embeddings=cfg.max_position_embeddings,
            rms_norm_eps=cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
            attention_bias=cfg.attention_bias,
            mlp_bias=cfg.mlp_bias,
            tie_word_embeddings=cfg.tie_word_embeddings,
        )
    config._attn_implementation = "eager"
    hf = LlamaForCausalLM(config).float()
    src = minimal.model
    dst = hf.model

    dst.embed_tokens.weight.data.copy_(src.embed_tokens.weight.data)
    dst.norm.weight.data.copy_(src.norm.weight.data)
    for dst_layer, src_layer in zip(dst.layers, src.layers):
        dst_layer.input_layernorm.weight.data.copy_(
            src_layer.input_layernorm.weight.data
        )
        dst_layer.post_attention_layernorm.weight.data.copy_(
            src_layer.post_attention_layernorm.weight.data
        )
        _export_attn_to_hf(src_layer.self_attn, dst_layer.self_attn, cfg)
        copy_linear(dst_layer.mlp.gate_proj, src_layer.mlp.gate_proj)
        copy_linear(dst_layer.mlp.up_proj, src_layer.mlp.up_proj)
        copy_linear(dst_layer.mlp.down_proj, src_layer.mlp.down_proj)

    hf.lm_head.weight.data.copy_(minimal.lm_head.weight.data)
    if cfg.tie_word_embeddings:
        hf.tie_weights()

    with torch.no_grad():
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in hf.state_dict().items()
        }


__all__ = [
    "export_llama_causal_to_hf_state_dict",
    "llama_config_from_hf",
    "load_hf_into_llama_causal",
]
