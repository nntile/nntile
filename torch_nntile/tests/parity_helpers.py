# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/parity_helpers.py
# Shared helpers for CPU / HF / nntile model submodule parity tests.

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor

from conftest import nntile_cpu


def assert_close(
    got: Tensor,
    ref: Tensor,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """Assert ``got`` (possibly on nntile) matches ``ref`` on CPU."""
    torch.testing.assert_close(
        nntile_cpu(got),
        ref.detach().cpu(),
        rtol=rtol,
        atol=atol,
    )


def clone_to_nntile(module: nn.Module) -> nn.Module:
    """Deep-copy ``module`` and move parameters/buffers to ``device=nntile``."""
    cloned = copy.deepcopy(module)
    with torch.no_grad():
        return cloned.to("nntile")


def rel_frobenius(a: Tensor, b: Tensor) -> float:
    """Relative Frobenius distance ``||a-b||_F / (||b||_F + eps)``."""
    a_cpu = nntile_cpu(a).float()
    b_cpu = nntile_cpu(b).float()
    diff = torch.linalg.norm(a_cpu - b_cpu)
    denom = torch.linalg.norm(b_cpu) + 1e-12
    return float((diff / denom).item())


def contiguous_to_nntile(tensor: Tensor) -> Tensor:
    """Ensure CPU contiguous layout before ``.to('nntile')``."""
    return tensor.detach().cpu().contiguous().to("nntile")


def copy_linear(dst: nn.Linear, src: nn.Linear) -> None:
    """Copy Linear weight (+ bias if both present)."""
    dst.weight.data.copy_(src.weight.data)
    if dst.bias is not None and src.bias is not None:
        dst.bias.data.copy_(src.bias.data)


def additive_causal_mask(batch: int, seq: int) -> Tensor:
    """Additive SDPA mask ``[B,1,S,S]`` with ``-inf`` above the diagonal."""
    mask = torch.zeros(batch, 1, seq, seq, dtype=torch.float32)
    mask.masked_fill_(
        torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return mask


def additive_local_causal_mask(
    batch: int,
    seq: int,
    window: int,
) -> Tensor:
    """GPT-Neo local attention additive mask ``[B,1,S,S]``.

    Allowed keys satisfy ``k <= q`` and ``q - k < window``.
    """
    q = torch.arange(seq).unsqueeze(1)
    k = torch.arange(seq).unsqueeze(0)
    allowed = (k <= q) & ((q - k) < window)
    mask = torch.zeros(seq, seq, dtype=torch.float32)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask.view(1, 1, seq, seq).expand(batch, 1, seq, seq).contiguous()


def assert_module_forward_backward(
    *,
    cpu_module: nn.Module,
    nnt_module: nn.Module,
    x_cpu: Tensor,
    forward_cpu,
    forward_nnt,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    bwd_rtol: float = 1e-3,
    bwd_atol: float = 1e-3,
) -> None:
    """Compare forward outputs and input grads for a submodule.

    ``forward_cpu(module, x)`` / ``forward_nnt(module, x_nnt)`` return the
    primary tensor output (logits / hidden).
    """
    x = x_cpu.detach().requires_grad_(True)
    y_ref = forward_cpu(cpu_module, x)
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)

    x_nnt = contiguous_to_nntile(x_cpu).requires_grad_(True)
    y_nnt = forward_nnt(nnt_module, x_nnt)
    assert_close(y_nnt, y_ref, rtol=rtol, atol=atol)
    (gx,) = torch.autograd.grad(
        y_nnt,
        x_nnt,
        grad_outputs=contiguous_to_nntile(grad),
    )
    assert_close(gx, x.grad, rtol=bwd_rtol, atol=bwd_atol)


def require_libnntile():
    """Import-time skip helper used by model parity modules."""
    import pytest

    from torch_nntile import _C

    return pytest.mark.skipif(
        not _C.has_libnntile(),
        reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
    )
