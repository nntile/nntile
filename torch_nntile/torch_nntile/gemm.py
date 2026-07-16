# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/gemm.py
# N-D GEMM matching ``nntile::tensor::gemm`` / C++ graph API semantics.

"""N-D GEMM for torch_nntile (not PyTorch last-2-dim matmul rules)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function

from torch_nntile import _C


class _NntileGemm(Function):
    @staticmethod
    def forward(
        ctx: Any,
        a: Tensor,
        b: Tensor,
        ndim: int,
        batch_ndim: int,
        trans_a: bool,
        trans_b: bool,
    ) -> Tensor:
        ctx.ndim = ndim
        ctx.batch_ndim = batch_ndim
        ctx.trans_a = bool(trans_a)
        ctx.trans_b = bool(trans_b)
        ctx.save_for_backward(a, b)
        return _C.gemm_forward(
            a, b, ndim, batch_ndim, ctx.trans_a, ctx.trans_b
        )

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor | None, ...]:
        a, b = ctx.saved_tensors
        grad_a, grad_b = _C.gemm_backward(
            a,
            b,
            grad_out,
            ctx.ndim,
            ctx.batch_ndim,
            [ctx.needs_input_grad[0], ctx.needs_input_grad[1]],
            ctx.trans_a,
            ctx.trans_b,
        )
        return grad_a, grad_b, None, None, None, None


def gemm(
    a: Tensor,
    b: Tensor,
    *,
    ndim: int,
    batch_ndim: int = 0,
    trans_a: bool = False,
    trans_b: bool = False,
) -> Tensor:
    """General N-D GEMM: ``C = op(A) @ op(B)`` with NNTile contraction semantics.

    ``trans_a`` / ``trans_b`` select NNTile transpose flags (swap the first
    ``ndim`` axes of the operand for the contraction), matching
    ``nntile::tensor::gemm`` - do not materialize ``Tensor.t()`` /
    ``swap_two_axes``.

    Examples (GPT-2 attention, ``batch_ndim=0``):

    - ``[B,S,H] @ [H,hs,n_heads] -> [B,S,hs,n_heads]`` with ``ndim=1``
    - ``[B,S,hs,n_heads] @ [hs,n_heads,H] -> [B,S,H]`` with ``ndim=2``
    - Linear / Conv1D-style ``[B,S,in] @ [out,in]^T`` via ``trans_b=True``
    """
    return _NntileGemm.apply(a, b, ndim, batch_ndim, trans_a, trans_b)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """``torch.matmul`` on nntile with inferred ``ndim`` / ``batch_ndim``."""
    return torch.matmul(a, b)


__all__ = ["gemm", "matmul"]
