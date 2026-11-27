# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/nn/add.py
"""Classic NNTile elementwise add for residuals."""

from __future__ import annotations

import torch
from torch import Tensor

from torch_nntile import _C


def add(
    x: Tensor,
    y: Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tensor:
    """``alpha * x + beta * y`` via classic ``tensor::add``."""
    if x.device.type != "nntile":
        return alpha * x + beta * y
    return _C.add(x, y, alpha, beta)


__all__ = ["add"]
