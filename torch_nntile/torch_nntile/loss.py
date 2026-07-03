# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/loss.py
# Cross-entropy loss support for device="nntile".

"""Loss helpers for the nntile device."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_ORIGINAL_CROSS_ENTROPY = F.cross_entropy


def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    size_average: bool | None = None,
    ignore_index: int = -100,
    reduce: bool | None = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy on ``device='nntile'`` via libnntile tensor ops."""
    if input.device.type != "nntile":
        return _ORIGINAL_CROSS_ENTROPY(
            input,
            target,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
    if weight is not None:
        raise ValueError("nntile cross_entropy does not support weight")
    if label_smoothing != 0.0:
        raise ValueError("nntile cross_entropy does not support label_smoothing")
    if reduction not in ("mean", "sum"):
        raise ValueError("nntile cross_entropy supports reduction 'mean' or 'sum'")
    from torch_nntile.training import cross_entropy as _nntile_cross_entropy

    return _nntile_cross_entropy(
        input,
        target,
        reduction=reduction,
        ignore_index=ignore_index,
    )


def patch_cross_entropy() -> None:
    """Route ``torch.nn.functional.cross_entropy`` to the nntile implementation."""
    F.cross_entropy = cross_entropy  # type: ignore[assignment]


patch_cross_entropy()

__all__ = ["cross_entropy", "patch_cross_entropy"]
