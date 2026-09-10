# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/nn/norm.py
"""Classic NNTile normalization for ``device=nntile``."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_nntile import _C
from torch_nntile.normalization import rms_norm as _rms_norm

_ORIGINAL_LAYER_NORM = F.layer_norm


def _as_normalized_shape(
    normalized_shape: int | Sequence[int],
) -> list[int]:
    if isinstance(normalized_shape, int):
        return [normalized_shape]
    return list(normalized_shape)


def layer_norm(
    input: Tensor,
    normalized_shape: int | Sequence[int],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Classic composed LayerNorm on ``device=nntile``."""
    if input.device.type != "nntile":
        return _ORIGINAL_LAYER_NORM(
            input,
            _as_normalized_shape(normalized_shape),
            weight,
            bias,
            eps,
        )
    return _C.layer_norm(
        input,
        _as_normalized_shape(normalized_shape),
        weight,
        bias,
        eps,
    )


def rms_norm(
    input: Tensor,
    normalized_shape: int | Sequence[int],
    weight: Tensor | None = None,
    eps: float | None = None,
) -> Tensor:
    """Classic RMSNorm (re-export from :mod:`torch_nntile.normalization`)."""
    return _rms_norm(input, normalized_shape, weight, eps)


__all__ = ["layer_norm", "rms_norm"]
