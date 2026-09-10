# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/dit_hf_loader.py
# Copy Diffusers DiTTransformer2DModel weights into torch_nntile DiT.

"""HF Diffusers DiT → NNTile DiT (AdaLN-Zero, non-overlapping patches)."""

from __future__ import annotations

from torch_nntile.models.dit import DiT, DiTConfig, _ADA_MOD_NAMES
from torch_nntile.models.hf_rope_layout import copy_linear
from torch_nntile.nn.linear import (
    linear_to_output_weight,
    linear_to_qkv_bias,
    linear_to_qkv_weight,
)


def dit_config_from_hf(hf_config) -> DiTConfig:
    out_ch = getattr(hf_config, "out_channels", None)
    return DiTConfig(
        sample_size=int(hf_config.sample_size),
        patch_size=int(hf_config.patch_size),
        in_channels=int(hf_config.in_channels),
        out_channels=None if out_ch is None else int(out_ch),
        num_layers=int(hf_config.num_layers),
        num_attention_heads=int(hf_config.num_attention_heads),
        attention_head_dim=int(hf_config.attention_head_dim),
        attention_bias=bool(hf_config.attention_bias),
        activation_fn=str(hf_config.activation_fn),
        num_embeds_ada_norm=int(hf_config.num_embeds_ada_norm),
        norm_eps=float(hf_config.norm_eps),
    )


def _load_qkv(dst, src_attn, n_heads: int, head_size: int) -> None:
    dst.query.weight.data.copy_(
        linear_to_qkv_weight(
            src_attn.to_q.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    dst.key.weight.data.copy_(
        linear_to_qkv_weight(
            src_attn.to_k.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    dst.value.weight.data.copy_(
        linear_to_qkv_weight(
            src_attn.to_v.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    if dst.query.bias is not None:
        dst.query.bias.data.copy_(
            linear_to_qkv_bias(
                src_attn.to_q.bias.data,
                n_heads=n_heads,
                head_size=head_size,
            )
        )
        dst.key.bias.data.copy_(
            linear_to_qkv_bias(
                src_attn.to_k.bias.data,
                n_heads=n_heads,
                head_size=head_size,
            )
        )
        dst.value.bias.data.copy_(
            linear_to_qkv_bias(
                src_attn.to_v.bias.data,
                n_heads=n_heads,
                head_size=head_size,
            )
        )
    dst.out.weight.data.copy_(
        linear_to_output_weight(
            src_attn.to_out[0].weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    if dst.out.bias is not None:
        dst.out.bias.data.copy_(src_attn.to_out[0].bias.data)


def _copy_fused_linear(src, dsts: list) -> None:
    """Split HF ``Linear(H, k*H)`` into ``k`` ``NntileLinear(H, H)``."""
    hidden = int(dsts[0].out_features)
    weight = src.weight.data
    bias = src.bias.data if src.bias is not None else None
    for i, dst in enumerate(dsts):
        dst.weight.data.copy_(weight[i * hidden : (i + 1) * hidden])
        if bias is not None and dst.bias is not None:
            dst.bias.data.copy_(bias[i * hidden : (i + 1) * hidden])


def _load_ada(dst, src_norm1) -> None:
    te = src_norm1.emb.timestep_embedder
    copy_linear(dst.emb.time_linear_1, te.linear_1)
    copy_linear(dst.emb.time_linear_2, te.linear_2)
    src_w = src_norm1.emb.class_embedder.embedding_table.weight.data
    n = min(dst.emb.class_embed.weight.shape[0], src_w.shape[0])
    dst.emb.class_embed.weight.data[:n].copy_(src_w[:n])
    _copy_fused_linear(
        src_norm1.linear,
        [getattr(dst, name) for name in _ADA_MOD_NAMES],
    )


def load_hf_into_dit(local: DiT, hf) -> None:
    """Copy Diffusers ``DiTTransformer2DModel`` weights into ``local``."""
    cfg = local.config
    conv_w = hf.pos_embed.proj.weight.data
    local.patch_embed.weight.data.copy_(
        conv_w.reshape(cfg.hidden_size, cfg.patch_dim).contiguous()
    )
    if hf.pos_embed.proj.bias is not None:
        local.patch_embed.bias.data.copy_(hf.pos_embed.proj.bias.data)
    pos = hf.pos_embed.pos_embed.data
    if pos.dim() == 3:
        pos = pos.squeeze(0)
    local.pos_embed.data.copy_(pos.contiguous())

    n_heads = cfg.num_attention_heads
    head_size = cfg.attention_head_dim
    for dst_block, src_block in zip(local.blocks, hf.transformer_blocks):
        _load_ada(dst_block.norm1, src_block.norm1)
        _load_qkv(dst_block.attn, src_block.attn1, n_heads, head_size)
        ff = src_block.ff.net
        copy_linear(dst_block.mlp.fc1, ff[0].proj)
        copy_linear(dst_block.mlp.fc2, ff[-1])

    copy_linear(local.proj_out_2, hf.proj_out_2)
    _copy_fused_linear(
        hf.proj_out_1,
        [local.proj_out_shift, local.proj_out_scale],
    )


__all__ = ["dit_config_from_hf", "load_hf_into_dit"]
