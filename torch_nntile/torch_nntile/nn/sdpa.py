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


class _NntileSdpaEager(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None,
        batch_ndim: int,
    ) -> Tensor:
        out = _C.sdpa_forward(q, k, v, mask, int(batch_ndim))
        ctx.save_for_backward(q, k, v)
        ctx.mask = mask
        ctx.batch_ndim = int(batch_ndim)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        q, k, v = ctx.saved_tensors
        grad_q, grad_k, grad_v = _C.sdpa_backward(
            q,
            k,
            v,
            grad_out,
            ctx.mask,
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
    """NNTile-layout SDPA eager on ``device='nntile'``.

    Q/K/V shape ``[batch..., seq, head_size]`` (C-order). Optional BOOL mask
    ``[q_seq, k_seq]`` on CPU or nntile (dim0 = query, dim1 = key, matching
    ``tensor::mask_scalar`` / GPT-2 ``attn_mask``). Scale is ``1/sqrt(head_size)``.
  """
    if q.device.type != "nntile":
        raise ValueError("sdpa_eager expects nntile Q/K/V tensors")
    if k.device.type != "nntile" or v.device.type != "nntile":
        raise ValueError("sdpa_eager expects nntile Q/K/V tensors")
    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        raise ValueError("nntile sdpa supports float32 only")
    if mask is not None and mask.dtype != torch.bool:
        raise ValueError("nntile sdpa: mask must be bool")
    return _NntileSdpaEager.apply(q, k, v, mask, batch_ndim)


class SDPA(nn.Module):
    """Scaled dot-product attention via libnntile ``sdpa_eager``."""

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


__all__ = ["SDPA", "sdpa_eager"]
