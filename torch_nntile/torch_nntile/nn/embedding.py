# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/nn/embedding.py
"""Classic NNTile embedding for ``device=nntile``."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_nntile import _C


def embedding(weight: Tensor, indices: Tensor) -> Tensor:
    """Classic ``tensor::embedding`` lookup (not ``aten::embedding``)."""
    if weight.device.type != "nntile":
        return F.embedding(indices, weight)
    return _C.nn_embedding(weight, indices)


__all__ = ["embedding"]
