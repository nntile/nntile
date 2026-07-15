# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_add_fiber_parity.py
# add_fiber parity vs broadcasted add (no scale_slice expand path).

from __future__ import annotations

import pytest
import torch

import torch_nntile
from torch_nntile import _C
from torch_nntile.add_fiber import add_fiber
from conftest import nntile_cpu

pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


def test_add_fiber_1d_bias_matches_broadcast_add():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    bias = torch.randn(8)
    ref = x + bias

    out = add_fiber(
        bias.to("nntile"),
        x.to("nntile"),
        axis=2,
        batch_ndim=0,
    )
    torch.testing.assert_close(nntile_cpu(out), ref, rtol=1e-5, atol=1e-5)


def test_add_fiber_qkv_bias_matches_broadcast_add():
    """C++ GPT-2 Q/K/V: fiber ``(n_heads, head_size)``, axis=3, batch_ndim=1."""
    torch.manual_seed(1)
    n_heads, batch, seq, hs = 4, 2, 8, 16
    x = torch.randn(n_heads, batch, seq, hs)
    bias = torch.randn(n_heads, hs)
    ref = x + bias.view(n_heads, 1, 1, hs)

    out = add_fiber(
        bias.to("nntile"),
        x.to("nntile"),
        axis=3,
        batch_ndim=1,
    )
    torch.testing.assert_close(nntile_cpu(out), ref, rtol=1e-5, atol=1e-5)


def test_add_fiber_backward_matches_broadcast_add():
    torch.manual_seed(2)
    n_heads, batch, seq, hs = 4, 2, 8, 16
    x_cpu = torch.randn(n_heads, batch, seq, hs, requires_grad=True)
    bias_cpu = torch.randn(n_heads, hs, requires_grad=True)
    grad_out = torch.randn_like(x_cpu)

    y_cpu = x_cpu + bias_cpu.view(n_heads, 1, 1, hs)
    y_cpu.backward(grad_out)

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    bias_nnt = bias_cpu.detach().to("nntile").requires_grad_(True)
    y_nnt = add_fiber(bias_nnt, x_nnt, axis=3, batch_ndim=1)
    gx, gb = torch.autograd.grad(
        y_nnt,
        (x_nnt, bias_nnt),
        grad_outputs=grad_out.to("nntile"),
    )
    torch.testing.assert_close(nntile_cpu(gx), x_cpu.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        nntile_cpu(gb), bias_cpu.grad, rtol=1e-4, atol=1e-4
    )
