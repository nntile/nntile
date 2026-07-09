# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_grad_accumulation.py
# Gradient accumulation parity: diamond fan-in, microbatch backward, grad.zero_.

from __future__ import annotations

import torch
import pytest

import torch_nntile
from torch_nntile import _C
from conftest import nntile_cpu


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


def test_diamond_shared_weight_grad_matches_cpu():
    """Weight used in two branches: w.grad must accumulate both paths."""
    torch.manual_seed(0)
    x_cpu = torch.randn(2, 3)
    w_cpu = torch.randn(4, 3, requires_grad=True)

    h1_cpu = torch.nn.functional.linear(x_cpu, w_cpu, None)
    h2_cpu = torch.nn.functional.linear(x_cpu, w_cpu, None)
    y_cpu = h1_cpu + h2_cpu
    y_cpu.backward(torch.ones_like(y_cpu))

    x_nnt = x_cpu.to("nntile")
    w_nnt = w_cpu.detach().to("nntile").requires_grad_(True)
    h1_nnt = torch.nn.functional.linear(x_nnt, w_nnt, None)
    h2_nnt = torch.nn.functional.linear(x_nnt, w_nnt, None)
    y_nnt = h1_nnt + h2_nnt
    y_nnt.backward(torch.ones(y_nnt.shape, device="cpu").to("nntile"))

    assert torch.allclose(nntile_cpu(w_nnt.grad), w_cpu.grad, rtol=1e-4, atol=1e-4)


def test_microbatch_grad_accumulation_matches_cpu():
    """Two backward() calls without clearing .grad exercises add_ on params."""
    torch.manual_seed(1)
    x1_cpu = torch.randn(2, 3)
    x2_cpu = torch.randn(2, 3)
    w_cpu = torch.randn(4, 3, requires_grad=True)

    y1_cpu = torch.nn.functional.relu(torch.nn.functional.linear(x1_cpu, w_cpu, None))
    y2_cpu = torch.nn.functional.relu(torch.nn.functional.linear(x2_cpu, w_cpu, None))
    grad_scale = 0.5
    y1_cpu.backward(torch.full_like(y1_cpu, grad_scale))
    y2_cpu.backward(torch.full_like(y2_cpu, grad_scale))

    w_ref = w_cpu.grad.clone()

    x1_nnt = x1_cpu.to("nntile")
    x2_nnt = x2_cpu.to("nntile")
    w_nnt = w_cpu.detach().to("nntile").requires_grad_(True)

    y1_nnt = torch.nn.functional.relu(
        torch.nn.functional.linear(x1_nnt, w_nnt, None)
    )
    y2_nnt = torch.nn.functional.relu(
        torch.nn.functional.linear(x2_nnt, w_nnt, None)
    )
    y1_nnt.backward(torch.full(y1_nnt.shape, grad_scale, device="cpu").to("nntile"))
    y2_nnt.backward(torch.full(y2_nnt.shape, grad_scale, device="cpu").to("nntile"))

    assert torch.allclose(nntile_cpu(w_nnt.grad), w_ref, rtol=1e-4, atol=1e-4)


def test_grad_zero_matches_cpu():
    """optimizer.zero_grad(set_to_none=False) path uses grad.zero_() -> fill_(0)."""
    grad_cpu = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    grad_nnt = grad_cpu.clone().to("nntile")

    grad_cpu.zero_()
    grad_nnt.zero_()

    assert torch.allclose(nntile_cpu(grad_nnt), grad_cpu)
    assert torch.all(nntile_cpu(grad_nnt) == 0)
