# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/nn/mul.py
"""Classic NNTile elementwise multiply."""

from __future__ import annotations

import torch
from torch import Tensor

from torch_nntile import _C


def mul(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise multiply via classic ``tensor::multiply``."""
    if a.device.type != "nntile":
        return a * b
    return _C.mul(a, b)


def mul_scalar(input: Tensor, scalar: float) -> Tensor:
    """``scalar * input`` via classic ``tensor::scale``."""
    if input.device.type != "nntile":
        return input * scalar
    return _C.mul_scalar(input, float(scalar))


__all__ = ["mul", "mul_scalar"]
