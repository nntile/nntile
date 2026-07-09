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


def vector_norm(
    input: torch.Tensor,
    ord: float | int = 2,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    out: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """2-norm on ``device='nntile'`` via libnntile tensor ops.

    Forward only: unlike ``F.layer_norm`` / ``F.rms_norm``, this does not
    register an autograd backward on nntile. Intended for use under
    ``torch.no_grad()`` (e.g. logging / clipping diagnostics). Raises if
    ``input.requires_grad`` and grad mode is enabled.
    """
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
        raise RuntimeError(
            "nntile linalg.vector_norm does not support dtype conversion"
        )
    if not _is_two_norm(ord):
        raise RuntimeError(
            "nntile linalg.vector_norm supports ord=2 only "
            "(no CPU round-trip)"
        )
    if input.requires_grad and torch.is_grad_enabled():
        raise RuntimeError(
            "nntile linalg.vector_norm is forward-only; call it under "
            "torch.no_grad() or detach the input. Use F.layer_norm / "
            "F.rms_norm when you need differentiable normalization."
        )
    axis = _normalize_dim(dim, input.ndim)
    if out is not None:
        _C.norm_forward(input, axis, keepdim, out)
        return out
    output, _norm_values = _C.norm_forward(input, axis, keepdim)
    return output


def patch_vector_norm() -> None:
    """Route ``torch.linalg.vector_norm`` to the nntile implementation."""
    torch.linalg.vector_norm = vector_norm  # type: ignore[assignment]


patch_vector_norm()

__all__ = ["vector_norm", "patch_vector_norm"]
