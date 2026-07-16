# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/sum_slice.py
# sum_slice matching ``nntile::tensor::sum_slice`` / old ``layer.gap.GAP``.

"""Sum over an axis into a slice (NNTile ``sum_slice``).

Old ``nntile.layer.gap.GAP`` forward is::

    alpha = 1 / x.shape[0]
    sum_slice(alpha, x, 0.0, yT, axis=0)   # → [batch, ...]
    transpose(yT → y)                       # side-R Linear layout

Here ``sum_slice`` / ``gap`` expose the reduction itself (``yT``). Torch-side
MLP-Mixer keeps ``[batch, channels]`` and applies a side-L classifier gemm.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function

from torch_nntile import _C


class _NntileSumSlice(Function):
    @staticmethod
    def forward(
        ctx: Any,
        src: Tensor,
        axis: int,
        alpha: float,
        beta: float,
    ) -> Tensor:
        ctx.axis = int(axis)
        ctx.alpha = float(alpha)
        ctx.save_for_backward(src)
        return _C.sum_slice_forward(src, ctx.axis, ctx.alpha, float(beta))

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor | None, ...]:
        (src,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        grad_src = _C.sum_slice_backward(
            grad_out, src, ctx.axis, ctx.alpha
        )
        return grad_src, None, None, None


def sum_slice(
    src: Tensor,
    *,
    axis: int,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Tensor:
    """``out = alpha * sum_slice(src, axis) + beta * out`` (beta must be 0)."""
    return _NntileSumSlice.apply(src, axis, alpha, beta)


def gap(x: Tensor) -> Tensor:
    """Global average pool over axis 0 (old ``GAP`` without side-R transpose)."""
    return sum_slice(x, axis=0, alpha=1.0 / float(x.shape[0]), beta=0.0)


__all__ = ["gap", "sum_slice"]
