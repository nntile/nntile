# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/gpt_neo_hf_loader.py
# Convert weights between HuggingFace GPT-Neo and torch_nntile GPTNeoCausal.

"""Bidirectional HF <-> NNTile weight conversion for GPT-Neo."""

from __future__ import annotations

import torch
from transformers import GPTNeoConfig as HfGPTNeoConfig
from transformers import GPTNeoForCausalLM

from torch_nntile.models.gpt_neo import GPTNeoCausal, GPTNeoConfig
from torch_nntile.models.hf_rope_layout import copy_linear
from torch_nntile.nn.linear import (
    linear_to_output_weight,
    linear_to_qkv_weight,
    output_to_linear_weight,
    qkv_to_linear_weight,
)


def gpt_neo_config_from_hf(hf: HfGPTNeoConfig) -> GPTNeoConfig:
    """Build a local ``GPTNeoConfig`` from an HF config."""
    intermediate = getattr(hf, "intermediate_size", None)
    if intermediate is None:
        intermediate = 4 * int(hf.hidden_size)
    attention_layers = list(getattr(hf, "attention_layers", []) or [])
    return GPTNeoConfig(
        vocab_size=int(hf.vocab_size),
        hidden_size=int(hf.hidden_size),
        intermediate_size=int(intermediate),
        num_hidden_layers=int(hf.num_layers),
        num_attention_heads=int(hf.num_heads),
        max_position_embeddings=int(hf.max_position_embeddings),
        window_size=int(getattr(hf, "window_size", 256)),
        layer_norm_eps=float(hf.layer_norm_epsilon),
        tie_word_embeddings=False,  # local models stay untied (migration debt)
        attention_layers=attention_layers,
    )


def _load_attn_from_hf(dst_attn, src_attn) -> None:
    inner = src_attn.attention
    n_heads = dst_attn.n_heads
    head_size = dst_attn.head_dim
    dst_attn.q_weight.data.copy_(
        linear_to_qkv_weight(
            inner.q_proj.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    dst_attn.k_weight.data.copy_(
        linear_to_qkv_weight(
            inner.k_proj.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    dst_attn.v_weight.data.copy_(
        linear_to_qkv_weight(
            inner.v_proj.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    dst_attn.o_weight.data.copy_(
        linear_to_output_weight(
            inner.out_proj.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    dst_attn.o_bias.data.copy_(inner.out_proj.bias.data)


def _export_attn_to_hf(src_attn, dst_attn) -> None:
    inner = dst_attn.attention
    inner.q_proj.weight.data.copy_(qkv_to_linear_weight(src_attn.q_weight.data))
    inner.k_proj.weight.data.copy_(qkv_to_linear_weight(src_attn.k_weight.data))
    inner.v_proj.weight.data.copy_(qkv_to_linear_weight(src_attn.v_weight.data))
    inner.out_proj.weight.data.copy_(
        output_to_linear_weight(src_attn.o_weight.data)
    )
    inner.out_proj.bias.data.copy_(src_attn.o_bias.data)


def load_hf_into_gpt_neo_causal(
    minimal: GPTNeoCausal,
    hf: GPTNeoForCausalLM,
) -> None:
    """Copy HF weights into ``minimal`` (CPU; call ``.to('nntile')`` after)."""
    src = hf.transformer
    dst = minimal.transformer

    dst.wte.weight.data.copy_(src.wte.weight.data)
    dst.wpe.weight.data.copy_(src.wpe.weight.data)
    dst.ln_f.weight.data.copy_(src.ln_f.weight.data)
    dst.ln_f.bias.data.copy_(src.ln_f.bias.data)

    for dst_block, src_block in zip(dst.h, src.h):
        dst_block.ln_1.weight.data.copy_(src_block.ln_1.weight.data)
        dst_block.ln_1.bias.data.copy_(src_block.ln_1.bias.data)
        dst_block.ln_2.weight.data.copy_(src_block.ln_2.weight.data)
        dst_block.ln_2.bias.data.copy_(src_block.ln_2.bias.data)
        _load_attn_from_hf(dst_block.attn, src_block.attn)
        copy_linear(dst_block.mlp.c_fc, src_block.mlp.c_fc)
        copy_linear(dst_block.mlp.c_proj, src_block.mlp.c_proj)

    # Always keep an independent lm_head (tying deferred; migration debt).
    minimal.lm_head.weight.data.copy_(hf.lm_head.weight.data)


def export_gpt_neo_causal_to_hf_state_dict(
    minimal: GPTNeoCausal,
    *,
    config: HfGPTNeoConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Export local GPT-Neo CPU weights as an HF state_dict."""
    cfg = minimal.config
    if config is None:
        config = HfGPTNeoConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            max_position_embeddings=cfg.max_position_embeddings,
            intermediate_size=cfg.intermediate_size,
            window_size=cfg.window_size,
            layer_norm_epsilon=cfg.layer_norm_eps,
            attention_layers=list(cfg.attention_layers),
            attention_types=[[["global", "local"], cfg.num_hidden_layers]],
            tie_word_embeddings=cfg.tie_word_embeddings,
            activation_function="gelu_new",
            attention_dropout=0.0,
            embed_dropout=0.0,
            resid_dropout=0.0,
        )
    config._attn_implementation = "eager"
    hf = GPTNeoForCausalLM(config).float()
    src = minimal.transformer
    dst = hf.transformer

    dst.wte.weight.data.copy_(src.wte.weight.data)
    dst.wpe.weight.data.copy_(src.wpe.weight.data)
    dst.ln_f.weight.data.copy_(src.ln_f.weight.data)
    dst.ln_f.bias.data.copy_(src.ln_f.bias.data)

    for dst_block, src_block in zip(dst.h, src.h):
        dst_block.ln_1.weight.data.copy_(src_block.ln_1.weight.data)
        dst_block.ln_1.bias.data.copy_(src_block.ln_1.bias.data)
        dst_block.ln_2.weight.data.copy_(src_block.ln_2.weight.data)
        dst_block.ln_2.bias.data.copy_(src_block.ln_2.bias.data)
        _export_attn_to_hf(src_block.attn, dst_block.attn)
        copy_linear(dst_block.mlp.c_fc, src_block.mlp.c_fc)
        copy_linear(dst_block.mlp.c_proj, src_block.mlp.c_proj)

    hf.lm_head.weight.data.copy_(minimal.lm_head.weight.data)
    # Keep lm_head untied (migration debt).

    with torch.no_grad():
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in hf.state_dict().items()
        }


__all__ = [
    "export_gpt_neo_causal_to_hf_state_dict",
    "gpt_neo_config_from_hf",
    "load_hf_into_gpt_neo_causal",
]
