# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/norm.py
# 2-norm support for device="nntile".

"""L2 norm helpers for the nntile device."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from torch_nntile import _C

_ORIGINAL_LINALG_VECTOR_NORM = torch.linalg.vector_norm


def _normalize_dim(dim: int | Sequence[int] | None, ndim: int) -> int | None:
    if dim is None:
        return None
    if isinstance(dim, int):
        axis = dim
    else:
        dims = list(dim)
        if len(dims) != 1:
            raise ValueError(
                "nntile linalg.vector_norm supports a single dim only"
            )
        axis = dims[0]
    if axis < 0:
        axis += ndim
    return axis


def _is_two_norm(ord: float | int) -> bool:
    if isinstance(ord, int):
        return ord == 2
    return abs(float(ord) - 2.0) < 1e-6


class _NntileVectorNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        ord: float | int,
        dim: int | None,
        keepdim: bool,
    ) -> torch.Tensor:
        output, norm_values = _C.norm_forward(input, dim, keepdim)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.save_for_backward(input, norm_values)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, norm_values = ctx.saved_tensors
        grad_input = _C.norm_backward(
            grad_output,
            input,
            norm_values,
            ctx.dim,
            ctx.keepdim,
        )
        return grad_input, None, None, None


def vector_norm(
    input: torch.Tensor,
    ord: float | int = 2,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    out: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """2-norm on ``device='nntile'`` via libnntile tensor ops."""
    if input.device.type != "nntile":
        return _ORIGINAL_LINALG_VECTOR_NORM(
            input,
            ord,
            dim,
            keepdim,
            out=out,
            dtype=dtype,
        )
    if out is not None and out.device.type != "nntile":
        raise RuntimeError("nntile vector_norm: out tensor must be on nntile")
    if dtype is not None:
        return _ORIGINAL_LINALG_VECTOR_NORM(
            input,
            ord,
            dim,
            keepdim,
            out=out,
            dtype=dtype,
        ).to(input.device)
    if not _is_two_norm(ord):
        cpu_out = _ORIGINAL_LINALG_VECTOR_NORM(
            input.cpu(),
            ord,
            dim,
            keepdim,
            out=None,
        )
        if out is not None:
            out.copy_(cpu_out.to(input.device))
            return out
        return cpu_out.to(input.device)
    axis = _normalize_dim(dim, input.ndim)
    if out is not None:
        if input.requires_grad:
            raise RuntimeError(
                "linalg_vector_norm(): functions with out=... arguments don't "
                "support automatic differentiation, but one of the arguments "
                "requires grad."
            )
        _C.norm_forward(input, axis, keepdim, out)
        return out
    return _NntileVectorNorm.apply(input, ord, axis, keepdim)


def patch_vector_norm() -> None:
    """Route ``torch.linalg.vector_norm`` to the nntile implementation."""
    torch.linalg.vector_norm = vector_norm  # type: ignore[assignment]


patch_vector_norm()

__all__ = ["vector_norm", "patch_vector_norm"]
