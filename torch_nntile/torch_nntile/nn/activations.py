# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/nn/activations.py
"""Classic NNTile activation kernels for ``device=nntile``."""

from __future__ import annotations

import torch
from torch import Tensor

from torch_nntile import _C


def relu(input: Tensor) -> Tensor:
    """Classic NNTile ReLU."""
    if input.device.type != "nntile":
        return torch.relu(input)
    return _C.relu(input)


def silu(input: Tensor) -> Tensor:
    """Classic NNTile SiLU."""
    if input.device.type != "nntile":
        return torch.nn.functional.silu(input)
    return _C.silu(input)


def gelu(input: Tensor, *, approximate: str = "tanh") -> Tensor:
    """Classic NNTile GELU (default: tanh approximation / gelutanh)."""
    if input.device.type != "nntile":
        return torch.nn.functional.gelu(input, approximate=approximate)
    return _C.gelu(input, approximate == "tanh")


__all__ = ["gelu", "relu", "silu"]
