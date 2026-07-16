# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/add_fiber.py
# add_fiber matching ``nntile::tensor::add_fiber`` / C++ GPT-2 bias path.

"""Add a fiber without materializing a broadcast (no ``scale_slice``)."""

from __future__ import annotations

from torch import Tensor

from torch_nntile import _C


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
    return _C.add_fiber(fiber, tensor, axis, batch_ndim, alpha, beta)


__all__ = ["add_fiber"]
