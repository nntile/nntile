# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/nn/weight_layout.py
# Attention weight layout conversion between PyTorch/HF and NNTile.

"""Convert attention projection weights for NNTile GEMM layouts."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

_QKV_KEYS = ("q_weight", "k_weight", "v_weight")
_O_KEY = "o_weight"
_BIAS_KEYS = ("q_bias", "k_bias", "v_bias", "o_bias")


def torch_to_nntile_qkv_weight(w: Tensor) -> Tensor:
    """``(hidden, n_heads, head_size)`` -> ``(hidden, head_size, n_heads)``."""
    if w.dim() != 3:
        raise ValueError("qkv weight must be 3D (hidden, n_heads, head_size)")
    return w.transpose(1, 2).contiguous()


def nntile_to_torch_qkv_weight(w: Tensor) -> Tensor:
    """``(hidden, head_size, n_heads)`` -> ``(hidden, n_heads, head_size)``."""
    if w.dim() != 3:
        raise ValueError("qkv weight must be 3D (hidden, head_size, n_heads)")
    return w.transpose(1, 2).contiguous()


def torch_to_nntile_o_weight(w: Tensor) -> Tensor:
    """``(n_heads, head_size, hidden)`` -> ``(head_size, n_heads, hidden)``."""
    if w.dim() != 3:
        raise ValueError("o weight must be 3D (n_heads, head_size, hidden)")
    return w.transpose(0, 1).contiguous()


def nntile_to_torch_o_weight(w: Tensor) -> Tensor:
    """``(head_size, n_heads, hidden)`` -> ``(n_heads, head_size, hidden)``."""
    if w.dim() != 3:
        raise ValueError("o weight must be 3D (head_size, n_heads, hidden)")
    return w.transpose(0, 1).contiguous()


def _with_prefix(prefix: str, key: str) -> str:
    if not prefix:
        return key
    if prefix.endswith("."):
        return f"{prefix}{key}"
    return f"{prefix}.{key}"


def _matches_suffix(full_key: str, suffix: str) -> bool:
    return full_key == suffix or full_key.endswith(f".{suffix}")


def convert_attn_weights(
    weights: dict[str, Tensor],
    direction: Literal["torch_to_nntile", "nntile_to_torch"],
    *,
    prefix: str = "",
) -> dict[str, Tensor]:
    """Convert q/k/v/o weights in a state-dict slice; biases are unchanged."""
    if direction == "torch_to_nntile":
        qkv_fn = torch_to_nntile_qkv_weight
        o_fn = torch_to_nntile_o_weight
    elif direction == "nntile_to_torch":
        qkv_fn = nntile_to_torch_qkv_weight
        o_fn = nntile_to_torch_o_weight
    else:
        raise ValueError(
            "direction must be 'torch_to_nntile' or 'nntile_to_torch'"
        )

    prefix_with_dot = _with_prefix(prefix, "") if prefix else ""

    out: dict[str, Tensor] = dict(weights)
    for full_key, tensor in weights.items():
        if prefix_with_dot and not full_key.startswith(prefix_with_dot):
            continue
        for suffix in _QKV_KEYS:
            if _matches_suffix(full_key, suffix):
                out[full_key] = qkv_fn(tensor)
                break
        else:
            if _matches_suffix(full_key, _O_KEY):
                out[full_key] = o_fn(tensor)
            elif any(_matches_suffix(full_key, key) for key in _BIAS_KEYS):
                out[full_key] = tensor.contiguous()

    return out


__all__ = [
    "convert_attn_weights",
    "nntile_to_torch_o_weight",
    "nntile_to_torch_qkv_weight",
    "torch_to_nntile_o_weight",
    "torch_to_nntile_qkv_weight",
]
