# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/normalization.py
# RMSNorm support for device="nntile".

"""Normalization helpers for the nntile device."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from torch_nntile import _C

_ORIGINAL_RMS_NORM = F.rms_norm


def _as_normalized_shape(
    normalized_shape: int | Sequence[int],
) -> list[int]:
    if isinstance(normalized_shape, int):
        return [normalized_shape]
    return list(normalized_shape)


class _NntileRmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        normalized_shape: list[int],
        weight: torch.Tensor | None,
        eps: float | None,
    ) -> torch.Tensor:
        output, rstd = _C.rms_norm_forward(
            input,
            normalized_shape,
            weight,
            eps,
        )
        ctx.has_weight = weight is not None
        ctx.save_for_backward(input, rstd, *((weight,) if weight is not None else ()))
        ctx.normalized_shape = normalized_shape
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, rstd, *weight_saved = ctx.saved_tensors
        weight = weight_saved[0] if ctx.has_weight else None
        grad_input, grad_weight = _C.rms_norm_backward(
            grad_output,
            input,
            ctx.normalized_shape,
            rstd,
            weight,
            [ctx.needs_input_grad[0], ctx.has_weight and ctx.needs_input_grad[2]],
        )
        return grad_input, None, grad_weight if ctx.has_weight else None, None


def rms_norm(
    input: torch.Tensor,
    normalized_shape: int | Sequence[int],
    weight: torch.Tensor | None = None,
    eps: float | None = None,
) -> torch.Tensor:
    """RMS normalization on ``device='nntile'`` via libnntile tensor ops."""
    if input.device.type != "nntile":
        return _ORIGINAL_RMS_NORM(input, normalized_shape, weight, eps)
    return _NntileRmsNorm.apply(
        input,
        _as_normalized_shape(normalized_shape),
        weight,
        eps,
    )


def patch_rms_norm() -> None:
    """Route ``torch.nn.functional.rms_norm`` to the nntile implementation."""
    F.rms_norm = rms_norm  # type: ignore[assignment]


__all__ = ["rms_norm", "patch_rms_norm"]
