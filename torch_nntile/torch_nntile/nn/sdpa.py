# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/nn/sdpa.py
# SDPA eager layer for device="nntile".

"""Scaled dot-product attention matching ``nntile::sdpa_eager``."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_nntile import _C


class _NntileModelTranspose(torch.autograd.Function):
    """``nntile::transpose(src, model_ndim)`` cyclic axis reordering.

    Matches ``nntile/src/nn/ops/transpose.cc`` model-code axis semantics.
    """

    @staticmethod
    def forward(ctx, x: Tensor, model_ndim: int) -> Tensor:
        ctx.model_ndim = int(model_ndim)
        return _C.model_transpose_forward(x, int(model_ndim))

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor, None]:
        return _C.model_transpose_backward(grad_out, ctx.model_ndim), None


def nntile_model_transpose(x: Tensor, model_ndim: int) -> Tensor:
    """Apply model-code transpose axis (storage order) on nntile tensors."""
    return _NntileModelTranspose.apply(x, model_ndim)


class _NntileSdpaKernel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor,
        batch_ndim: int,
    ) -> Tensor:
        out = _C.sdpa_forward(q, k, v, mask, int(batch_ndim))
        ctx.save_for_backward(q, k, v, mask)
        ctx.batch_ndim = int(batch_ndim)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        q, k, v, mask = ctx.saved_tensors
        grad_q, grad_k, grad_v = _C.sdpa_backward(
            q,
            k,
            v,
            grad_out,
            mask,
            ctx.batch_ndim,
        )
        return grad_q, grad_k, grad_v, None, None


def sdpa_eager(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    *,
    batch_ndim: int = 2,
) -> Tensor:
    """SDPA on post-GEMM Q/K/V layout ``[batch, seq, head_size, n_heads]``.

    Internally applies ``transpose(..., 1)`` to kernel layout (e.g.
    ``[n_heads, batch, seq, head_size]`` when ``batch_ndim=2``), calls
    ``F.scaled_dot_product_attention`` (dispatched to the nntile ATen
    backend), then ``transpose(..., 3)`` on the output — matching
    ``nntile/src/model/gpt2/gpt2_attention.cc`` around ``sdpa_eager``.
    Optional BOOL mask ``[q_seq, k_seq]`` (dim0 = query, dim1 = key). Scale is
    ``1/sqrt(head_size)``.
    """
    if batch_ndim != 2:
        raise ValueError("sdpa_eager currently supports batch_ndim=2 only")
    if q.device.type != "nntile":
        raise ValueError("sdpa_eager expects nntile Q/K/V tensors")
    if k.device.type != "nntile" or v.device.type != "nntile":
        raise ValueError("sdpa_eager expects nntile Q/K/V tensors")
    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        raise ValueError("nntile sdpa supports float32 only")
    if mask is not None and mask.dtype != torch.bool:
        raise ValueError("nntile sdpa: mask must be bool")
    if mask is not None and mask.device.type != "nntile":
        raise ValueError("nntile sdpa: mask must be on device nntile")
    q_sdpa = nntile_model_transpose(q, 1)
    k_sdpa = nntile_model_transpose(k, 1)
    v_sdpa = nntile_model_transpose(v, 1)
    if mask is None:
        attn_out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        )
    else:
        attn_out = _NntileSdpaKernel.apply(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            mask,
            batch_ndim,
        )
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


__all__ = ["SDPA", "nntile_model_transpose", "sdpa_eager"]
