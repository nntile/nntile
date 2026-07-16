# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/nn/sdpa.py
# SDPA eager layer for device="nntile".

"""Scaled dot-product attention matching ``nntile::sdpa_eager``."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from torch_nntile import _C


def nntile_model_transpose(x: Tensor, model_ndim: int) -> Tensor:
    """Apply model-code transpose axis (storage order) on nntile tensors."""
    return _C.model_transpose(x, model_ndim)


def _validate_sdpa_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None,
    *,
    batch_ndim: int,
) -> None:
    if batch_ndim not in (2, 3):
        raise ValueError("sdpa currently supports batch_ndim=2 or 3")
    if q.device.type != "nntile":
        raise ValueError("sdpa expects nntile Q/K/V tensors")
    if k.device.type != "nntile" or v.device.type != "nntile":
        raise ValueError("sdpa expects nntile Q/K/V tensors")
    if (
        q.dtype != torch.float32
        or k.dtype != torch.float32
        or v.dtype != torch.float32
    ):
        raise ValueError("nntile sdpa supports float32 only")
    if mask is not None and mask.dtype != torch.bool:
        raise ValueError("nntile sdpa: mask must be bool")
    if mask is not None and mask.device.type != "nntile":
        raise ValueError("nntile sdpa: mask must be on device nntile")


def sdpa_kernel(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    *,
    batch_ndim: int = 2,
) -> Tensor:
    """SDPA on kernel layout (matches ``nntile::sdpa_eager`` inputs).

    Expects ``[n_heads, batch, seq, head_size]`` when ``batch_ndim=2``.
    GQA callers may use ``batch_ndim=3`` with
    ``[n_kv_heads, n_rep, batch, seq, head_size]``.
    Optional BOOL mask ``[q_seq, k_seq]`` (dim0 = query, dim1 = key).
    ``mask=None`` means fully dense attention (still via libnntile, not
    ``F.scaled_dot_product_attention``, so GQA 5-D layouts work).
    """
    _validate_sdpa_inputs(q, k, v, mask, batch_ndim=batch_ndim)
    return _C.sdpa_kernel(q, k, v, mask, batch_ndim)


def sdpa_eager(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    *,
    batch_ndim: int = 2,
) -> Tensor:
    """SDPA on post-GEMM Q/K/V layout ``[batch, seq, head_size, n_heads]``.

    Applies ``transpose(..., 1)`` to kernel layout, runs ``sdpa_kernel``, then
    ``transpose(..., 3)`` on the output. Q/K/V biases (C++ ``add_fiber`` after
    transpose) belong in the caller - see ``GPT2Attention``.
    Optional BOOL mask ``[q_seq, k_seq]`` (dim0 = query, dim1 = key). Scale is
    ``1/sqrt(head_size)``.
    """
    _validate_sdpa_inputs(q, k, v, mask, batch_ndim=batch_ndim)
    q_sdpa = nntile_model_transpose(q, 1)
    k_sdpa = nntile_model_transpose(k, 1)
    v_sdpa = nntile_model_transpose(v, 1)
    attn_out = sdpa_kernel(q_sdpa, k_sdpa, v_sdpa, mask, batch_ndim=batch_ndim)
    return nntile_model_transpose(attn_out, 3)


class SDPA(nn.Module):
    """Scaled dot-product attention for post-GEMM Q/K/V tensors.

    Accepts Q/K/V in projection layout ``[batch, seq, head_size, n_heads]``.
    Delegates to ``sdpa_eager`` (transposes are handled there).
    """

    def __init__(self, batch_ndim: int = 2) -> None:
        super().__init__()
        self.batch_ndim = int(batch_ndim)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        return sdpa_eager(q, k, v, mask, batch_ndim=self.batch_ndim)


__all__ = ["SDPA", "nntile_model_transpose", "sdpa_eager", "sdpa_kernel"]
