# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/rope.py
# Rotary position embedding (RoPE) for device="nntile".

"""RoPE: ``y = rope(sin, cos, x)``.

Matches ``nntile::rope`` / ``NNRopeOp``. ``sin`` and ``cos`` share shape
``[..., head_dim // 2]``; ``x`` has the same leading dims with last axis
``head_dim`` (interleaved pairs). Only ``x`` receives gradients.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function

from torch_nntile import _C


def _rope_ref_forward(sin: Tensor, cos: Tensor, x: Tensor) -> Tensor:
    """Pure-torch RoPE matching the NNTile interleaved-pair kernel."""
    # x: [..., 2*m], sin/cos: [..., m] (broadcast over trailing dims of x).
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    y_even = cos * x_even - sin * x_odd
    y_odd = sin * x_even + cos * x_odd
    y = torch.empty_like(x)
    y[..., 0::2] = y_even
    y[..., 1::2] = y_odd
    return y


def _rope_ref_backward(sin: Tensor, cos: Tensor, grad_y: Tensor) -> Tensor:
    g_even = grad_y[..., 0::2]
    g_odd = grad_y[..., 1::2]
    dx_even = cos * g_even + sin * g_odd
    dx_odd = cos * g_odd - sin * g_even
    dx = torch.empty_like(grad_y)
    dx[..., 0::2] = dx_even
    dx[..., 1::2] = dx_odd
    return dx


class _RefRope(Function):
    """CPU / non-nntile autograd fallback (same math as NNTile kernel)."""

    @staticmethod
    def forward(ctx: Any, sin: Tensor, cos: Tensor, x: Tensor) -> Tensor:
        ctx.save_for_backward(sin, cos)
        return _rope_ref_forward(sin, cos, x)

    @staticmethod
    def backward(
        ctx: Any, grad_y: Tensor
    ) -> tuple[None, None, Tensor]:
        sin, cos = ctx.saved_tensors
        return None, None, _rope_ref_backward(sin, cos, grad_y)


def rope(sin: Tensor, cos: Tensor, x: Tensor) -> Tensor:
    """Apply rotary embeddings: ``y = rope(sin, cos, x)``.

    On ``device='nntile'`` uses the libnntile kernel via ``_C.rope``.
    Otherwise falls back to a pure-torch reference that matches the
    interleaved-pair layout.
    """
    if x.device.type == "nntile":
        return _C.rope(sin, cos, x)
    return _RefRope.apply(sin, cos, x)


def rope_sin_cos_from_position_ids(
    position_ids: Tensor,
    head_dim: int,
    *,
    rope_theta: float = 10000.0,
) -> tuple[Tensor, Tensor]:
    """Build ``(sin, cos)`` with shape ``[batch, seq, head_dim // 2]``.

    Mirrors HuggingFace default Llama RoPE
    (``_compute_default_rope_parameters``).
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    device = position_ids.device
    dtype = torch.float32
    half = head_dim // 2
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, half, dtype=dtype, device="cpu")
            .float()
            / float(half)
        )
    )
    if device.type != "cpu":
        inv_freq = inv_freq.to(device)
    # position_ids: [batch, seq]
    freqs = position_ids.to(dtype).unsqueeze(-1) * inv_freq.view(1, 1, -1)
    return freqs.sin(), freqs.cos()


__all__ = ["rope", "rope_sin_cos_from_position_ids"]
