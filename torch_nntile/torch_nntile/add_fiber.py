# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/add_fiber.py
# add_fiber matching ``nntile::tensor::add_fiber`` / C++ GPT-2 bias path.

"""Add a fiber without materializing a broadcast (no ``scale_slice``)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function

from torch_nntile import _C


class _NntileAddFiber(Function):
    @staticmethod
    def forward(
        ctx: Any,
        fiber: Tensor,
        tensor: Tensor,
        axis: int,
        batch_ndim: int,
        alpha: float,
        beta: float,
    ) -> Tensor:
        ctx.axis = int(axis)
        ctx.batch_ndim = int(batch_ndim)
        ctx.alpha = float(alpha)
        ctx.beta = float(beta)
        ctx.save_for_backward(fiber, tensor)
        return _C.add_fiber_forward(
            fiber, tensor, ctx.axis, ctx.batch_ndim, ctx.alpha, ctx.beta
        )

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor | None, ...]:
        fiber, tensor = ctx.saved_tensors
        grad_fiber, grad_tensor = _C.add_fiber_backward(
            grad_out,
            fiber,
            tensor,
            ctx.axis,
            ctx.batch_ndim,
            [ctx.needs_input_grad[0], ctx.needs_input_grad[1]],
            ctx.alpha,
            ctx.beta,
        )
        return grad_fiber, grad_tensor, None, None, None, None


def add_fiber(
    fiber: Tensor,
    tensor: Tensor,
    *,
    axis: int,
    batch_ndim: int = 0,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tensor:
    """``out = alpha * fiber + beta * tensor`` along fibers (C++ ``add_fiber``).

    ``fiber`` has shape ``tensor.shape[:batch_ndim] + (tensor.shape[axis],)``.
    Unlike ``tensor + fiber.view(...)``, this does not expand the fiber via
    ``scale_slice`` / broadcast.
    """
    return _NntileAddFiber.apply(
        fiber, tensor, axis, batch_ndim, alpha, beta
    )


__all__ = ["add_fiber"]
