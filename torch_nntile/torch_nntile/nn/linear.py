# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/nn/linear.py
# Linear / attention projection modules for nntile models.

"""Linear and attention projection helpers matching deleted NNGraph layouts."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from torch_nntile.nn.functional import add_fiber, gemm
from torch_nntile.models.gpt2_minimal import make_causal_sdpa_mask
from torch_nntile.nn.sdpa import nntile_model_transpose


class NntileLinear(nn.Module):
    """Linear layer via ``gemm(..., trans_b=True)`` and ``add_fiber``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        out = gemm(
            x,
            self.weight,
            ndim=1,
            batch_ndim=0,
            trans_a=False,
            trans_b=True,
        )
        if self.bias is not None:
            out = add_fiber(self.bias, out, axis=out.dim() - 1, batch_ndim=0)
        return out


class NntileQKVProjection(nn.Module):
    """Q/K/V projection weight ``[hidden, head_size, n_heads]``."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        n_heads: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.n_heads = n_heads
        self.weight = nn.Parameter(
            torch.empty(hidden_size, head_size, n_heads)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(n_heads, head_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        proj = gemm(x, self.weight, ndim=1, batch_ndim=0)
        out = nntile_model_transpose(proj, 1)
        if self.bias is not None:
            out = add_fiber(self.bias, out, axis=3, batch_ndim=1)
        return out


class NntileAttentionOutput(nn.Module):
    """Attention O projection weight ``[head_size, n_heads, hidden]``."""

    def __init__(
        self,
        head_size: int,
        n_heads: int,
        hidden_size: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(
            torch.empty(head_size, n_heads, hidden_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, attn_out: Tensor) -> Tensor:
        attn_t = nntile_model_transpose(attn_out, 3)
        out = gemm(attn_t, self.weight, ndim=2, batch_ndim=0)
        if self.bias is not None:
            out = add_fiber(self.bias, out, axis=out.dim() - 1, batch_ndim=0)
        return out


def linear_to_qkv_weight(
    weight: Tensor,
    *,
    n_heads: int,
    head_size: int,
) -> Tensor:
    """HF ``[heads * head_size, in]`` to ``[in, head_size, heads]``."""
    return (
        weight.detach()
        .reshape(n_heads, head_size, -1)
        .permute(2, 1, 0)
        .contiguous()
    )


def qkv_to_linear_weight(weight: Tensor) -> Tensor:
    """``[in, head_size, heads]`` to HF ``[heads * head_size, in]``."""
    hidden, head_size, n_heads = weight.shape
    del hidden
    return weight.detach().permute(2, 1, 0).reshape(n_heads * head_size, -1)


def linear_to_qkv_bias(
    bias: Tensor,
    *,
    n_heads: int,
    head_size: int,
) -> Tensor:
    """HF ``[heads * head_size]`` to ``[heads, head_size]``."""
    return bias.detach().reshape(n_heads, head_size).contiguous()


def qkv_to_linear_bias(bias: Tensor) -> Tensor:
    """``[heads, head_size]`` to HF ``[heads * head_size]``."""
    return bias.detach().reshape(-1).contiguous()


def linear_to_output_weight(
    weight: Tensor,
    *,
    n_heads: int,
    head_size: int,
) -> Tensor:
    """HF O ``[out, heads * head_size]`` to ``[head_size, heads, out]``."""
    out_features = weight.shape[0]
    return (
        weight.detach()
        .reshape(out_features, n_heads, head_size)
        .permute(2, 1, 0)
        .contiguous()
    )


def output_to_linear_weight(weight: Tensor) -> Tensor:
    """``[head_size, heads, out]`` to HF O ``[out, heads * head_size]``."""
    head_size, n_heads, out_features = weight.shape
    return (
        weight.detach()
        .permute(2, 1, 0)
        .reshape(out_features, n_heads * head_size)
        .contiguous()
    )


def linear_to_gqa_q_weight(
    weight: Tensor,
    *,
    n_kv_heads: int,
    n_rep: int,
    head_size: int,
) -> Tensor:
    """HF Q ``[n_kv * n_rep * head_size, in]`` to 4D NNGraph layout."""
    return (
        weight.detach()
        .reshape(n_kv_heads, n_rep, head_size, -1)
        .permute(3, 2, 0, 1)
        .contiguous()
    )


def gqa_q_to_linear_weight(weight: Tensor) -> Tensor:
    """4D NNGraph Q layout to HF Q ``[heads * head_size, in]``."""
    hidden, head_size, n_kv_heads, n_rep = weight.shape
    del hidden
    return (
        weight.detach()
        .permute(2, 3, 1, 0)
        .reshape(n_kv_heads * n_rep * head_size, -1)
        .contiguous()
    )


def linear_to_gqa_output_weight(
    weight: Tensor,
    *,
    n_kv_heads: int,
    n_rep: int,
    head_size: int,
) -> Tensor:
    """HF O ``[out, n_kv * n_rep * head_size]`` to 4D NNGraph layout."""
    out_features = weight.shape[0]
    return (
        weight.detach()
        .reshape(out_features, n_kv_heads, n_rep, head_size)
        .permute(3, 1, 2, 0)
        .contiguous()
    )


def gqa_output_to_linear_weight(weight: Tensor) -> Tensor:
    """4D NNGraph O layout to HF O ``[out, heads * head_size]``."""
    head_size, n_kv_heads, n_rep, out_features = weight.shape
    return (
        weight.detach()
        .permute(3, 1, 2, 0)
        .reshape(out_features, n_kv_heads * n_rep * head_size)
        .contiguous()
    )


def prepare_sdpa_mask(
    mask: Tensor | None,
    x: Tensor,
    *,
    q_len: int,
    k_len: int | None = None,
    is_causal: bool = False,
) -> Tensor | None:
    """Return a BOOL ``[q, k]`` mask on ``x.device`` for NNGraph SDPA."""
    if k_len is None:
        k_len = q_len
    if mask is None and is_causal:
        mask = make_causal_sdpa_mask(q_len, device=x.device)
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        bool_mask = mask
    else:
        bool_mask = torch.isfinite(mask) & (mask >= 0)
    while bool_mask.dim() > 2:
        bool_mask = bool_mask.select(0, 0)
    if bool_mask.shape != (q_len, k_len):
        bool_mask = bool_mask.reshape(q_len, k_len)
    if bool_mask.device != x.device:
        bool_mask = bool_mask.to(x.device)
    return bool_mask.contiguous()


__all__ = [
    "NntileAttentionOutput",
    "NntileLinear",
    "NntileQKVProjection",
    "gqa_output_to_linear_weight",
    "gqa_q_to_linear_weight",
    "linear_to_gqa_output_weight",
    "linear_to_gqa_q_weight",
    "linear_to_output_weight",
    "linear_to_qkv_bias",
    "linear_to_qkv_weight",
    "output_to_linear_weight",
    "prepare_sdpa_mask",
    "qkv_to_linear_bias",
    "qkv_to_linear_weight",
]
