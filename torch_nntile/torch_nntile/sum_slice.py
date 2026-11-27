# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/sum_slice.py
# sum_slice matching ``nntile::tensor::sum_slice`` / old ``layer.gap.GAP``.

"""Sum over an axis into a slice (NNTile ``sum_slice``).

Old ``nntile.layer.gap.GAP`` forward is::

    alpha = 1 / x.shape[0]
    sum_slice(alpha, x, 0.0, yT, axis=0)   # -> [batch, ...]
    transpose(yT -> y)                       # side-R Linear layout

Here ``sum_slice`` / ``gap`` expose the reduction itself (``yT``). Torch-side
MLP-Mixer keeps ``[batch, channels]`` and applies a side-L classifier gemm.
"""

from __future__ import annotations

from torch import Tensor

from torch_nntile import _C


def sum_slice(
    src: Tensor,
    *,
    axis: int,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Tensor:
    """``out = alpha * sum_slice(src, axis) + beta * out`` (beta must be 0)."""
    return _C.sum_slice(src, axis, alpha, beta)


def gap(x: Tensor) -> Tensor:
    """Global average pool over axis 0 (old ``GAP`` without side-R transpose)."""
    return _C.gap(x)


__all__ = ["gap", "sum_slice"]
