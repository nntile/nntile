# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/hf_rope_layout.py
# Convert HF rotate_half RoPE channel order ↔ NNTile interleaved pairs.

"""RoPE weight-channel layout helpers for HF ↔ NNTile conversion.

HuggingFace Llama / GPT-NeoX apply ``rotate_half`` (first/second half of the
rotary dims). NNTile ``rope`` uses interleaved even/odd pairs. Matching
forward values requires rearranging Q/K (and fused QKV) output channels when
loading or exporting weights — the same transform used by the deleted
``nntile/tests/model/*/generate_test_data.py`` fixtures.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _interleave_along_head_dim(w: Tensor, rotary_elems: int) -> Tensor:
    """``(n_heads, head_dim, ...)`` → interleaved pairs on first rotary elems."""
    if rotary_elems <= 0:
        return w
    if rotary_elems % 2 != 0:
        raise ValueError("rotary_elems must be even for RoPE interleave")
    mid = rotary_elems // 2
    out = w.clone()
    selected = w[:, :rotary_elems, ...]
    y = torch.empty_like(selected)
    y[:, 0::2, ...] = selected[:, :mid, ...]
    y[:, 1::2, ...] = selected[:, mid:, ...]
    out[:, :rotary_elems, ...] = y
    return out


def _deinterleave_along_head_dim(w: Tensor, rotary_elems: int) -> Tensor:
    """Inverse of :func:`_interleave_along_head_dim`."""
    if rotary_elems <= 0:
        return w
    if rotary_elems % 2 != 0:
        raise ValueError("rotary_elems must be even for RoPE deinterleave")
    mid = rotary_elems // 2
    out = w.clone()
    selected = w[:, :rotary_elems, ...]
    y = torch.empty_like(selected)
    y[:, :mid, ...] = selected[:, 0::2, ...]
    y[:, mid:, ...] = selected[:, 1::2, ...]
    out[:, :rotary_elems, ...] = y
    return out


def hf_to_nntile_qkv_weight(
    weight: Tensor,
    *,
    n_heads: int,
    head_dim: int,
    rotary_pct: float = 1.0,
) -> Tensor:
    """HF ``Linear`` ``(n_heads*head_dim, in)`` → NNTile interleaved out rows."""
    out_f, in_f = weight.shape
    if out_f != n_heads * head_dim:
        raise ValueError(
            f"expected out_features={n_heads * head_dim}, got {out_f}"
        )
    rotary = int(head_dim * rotary_pct)
    w = weight.detach().reshape(n_heads, head_dim, in_f)
    w = _interleave_along_head_dim(w, rotary)
    return w.reshape(out_f, in_f).contiguous()


def nntile_to_hf_qkv_weight(
    weight: Tensor,
    *,
    n_heads: int,
    head_dim: int,
    rotary_pct: float = 1.0,
) -> Tensor:
    """NNTile interleaved ``(n_heads*head_dim, in)`` → HF ``rotate_half`` rows."""
    out_f, in_f = weight.shape
    if out_f != n_heads * head_dim:
        raise ValueError(
            f"expected out_features={n_heads * head_dim}, got {out_f}"
        )
    rotary = int(head_dim * rotary_pct)
    w = weight.detach().reshape(n_heads, head_dim, in_f)
    w = _deinterleave_along_head_dim(w, rotary)
    return w.reshape(out_f, in_f).contiguous()


def hf_to_nntile_qkv_bias(
    bias: Tensor,
    *,
    n_heads: int,
    head_dim: int,
    rotary_pct: float = 1.0,
) -> Tensor:
    """Same channel reorder for a Q or K bias vector."""
    if bias.numel() != n_heads * head_dim:
        raise ValueError(
            f"expected bias numel={n_heads * head_dim}, got {bias.numel()}"
        )
    rotary = int(head_dim * rotary_pct)
    b = bias.detach().reshape(n_heads, head_dim, 1)
    b = _interleave_along_head_dim(b, rotary)
    return b.reshape(-1).contiguous()


def nntile_to_hf_qkv_bias(
    bias: Tensor,
    *,
    n_heads: int,
    head_dim: int,
    rotary_pct: float = 1.0,
) -> Tensor:
    """Inverse of :func:`hf_to_nntile_qkv_bias`."""
    if bias.numel() != n_heads * head_dim:
        raise ValueError(
            f"expected bias numel={n_heads * head_dim}, got {bias.numel()}"
        )
    rotary = int(head_dim * rotary_pct)
    b = bias.detach().reshape(n_heads, head_dim, 1)
    b = _deinterleave_along_head_dim(b, rotary)
    return b.reshape(-1).contiguous()


def hf_to_nntile_fused_qkv_weight(
    weight: Tensor,
    *,
    n_heads: int,
    head_dim: int,
    rotary_pct: float = 1.0,
) -> Tensor:
    """HF GPT-NeoX ``query_key_value`` weight → interleaved Q/K channels.

    Weight layout is ``(n_heads, 3*head_dim, in)`` with per-head ``[q|k|v]``.
    """
    out_f, in_f = weight.shape
    hidden = n_heads * head_dim
    if out_f != 3 * hidden:
        raise ValueError(
            f"expected fused out_features={3 * hidden}, got {out_f}"
        )
    rotary = int(head_dim * rotary_pct)
    w = weight.detach().reshape(n_heads, 3 * head_dim, in_f).clone()
    q = _interleave_along_head_dim(w[:, :head_dim, :], rotary)
    k = _interleave_along_head_dim(w[:, head_dim : 2 * head_dim, :], rotary)
    v = w[:, 2 * head_dim : 3 * head_dim, :]
    fused = torch.cat([q, k, v], dim=1)
    return fused.reshape(out_f, in_f).contiguous()


def nntile_to_hf_fused_qkv_weight(
    weight: Tensor,
    *,
    n_heads: int,
    head_dim: int,
    rotary_pct: float = 1.0,
) -> Tensor:
    """Inverse of :func:`hf_to_nntile_fused_qkv_weight`."""
    out_f, in_f = weight.shape
    hidden = n_heads * head_dim
    if out_f != 3 * hidden:
        raise ValueError(
            f"expected fused out_features={3 * hidden}, got {out_f}"
        )
    rotary = int(head_dim * rotary_pct)
    w = weight.detach().reshape(n_heads, 3 * head_dim, in_f).clone()
    q = _deinterleave_along_head_dim(w[:, :head_dim, :], rotary)
    k = _deinterleave_along_head_dim(
        w[:, head_dim : 2 * head_dim, :], rotary
    )
    v = w[:, 2 * head_dim : 3 * head_dim, :]
    fused = torch.cat([q, k, v], dim=1)
    return fused.reshape(out_f, in_f).contiguous()


def hf_to_nntile_fused_qkv_bias(
    bias: Tensor,
    *,
    n_heads: int,
    head_dim: int,
    rotary_pct: float = 1.0,
) -> Tensor:
    """HF GPT-NeoX fused QKV bias → interleaved Q/K channels."""
    hidden = n_heads * head_dim
    if bias.numel() != 3 * hidden:
        raise ValueError(
            f"expected fused bias numel={3 * hidden}, got {bias.numel()}"
        )
    rotary = int(head_dim * rotary_pct)
    b = bias.detach().reshape(n_heads, 3 * head_dim, 1).clone()
    q = _interleave_along_head_dim(b[:, :head_dim, :], rotary)
    k = _interleave_along_head_dim(b[:, head_dim : 2 * head_dim, :], rotary)
    v = b[:, 2 * head_dim : 3 * head_dim, :]
    return torch.cat([q, k, v], dim=1).reshape(-1).contiguous()


def nntile_to_hf_fused_qkv_bias(
    bias: Tensor,
    *,
    n_heads: int,
    head_dim: int,
    rotary_pct: float = 1.0,
) -> Tensor:
    """Inverse of :func:`hf_to_nntile_fused_qkv_bias`."""
    hidden = n_heads * head_dim
    if bias.numel() != 3 * hidden:
        raise ValueError(
            f"expected fused bias numel={3 * hidden}, got {bias.numel()}"
        )
    rotary = int(head_dim * rotary_pct)
    b = bias.detach().reshape(n_heads, 3 * head_dim, 1).clone()
    q = _deinterleave_along_head_dim(b[:, :head_dim, :], rotary)
    k = _deinterleave_along_head_dim(
        b[:, head_dim : 2 * head_dim, :], rotary
    )
    v = b[:, 2 * head_dim : 3 * head_dim, :]
    return torch.cat([q, k, v], dim=1).reshape(-1).contiguous()


def copy_linear(dst: torch.nn.Linear, src: torch.nn.Linear) -> None:
    """Copy ``src`` weights (and bias if both present) into ``dst``."""
    dst.weight.data.copy_(src.weight.data)
    if dst.bias is not None and src.bias is not None:
        dst.bias.data.copy_(src.bias.data)


__all__ = [
    "copy_linear",
    "hf_to_nntile_fused_qkv_bias",
    "hf_to_nntile_fused_qkv_weight",
    "hf_to_nntile_qkv_bias",
    "hf_to_nntile_qkv_weight",
    "nntile_to_hf_fused_qkv_bias",
    "nntile_to_hf_fused_qkv_weight",
    "nntile_to_hf_qkv_bias",
    "nntile_to_hf_qkv_weight",
]
