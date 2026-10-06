# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_mse_loss_parity.py
# mse_loss parity: CPU PyTorch vs nntile tensor ops, with upstream
# grad_output scaling (composed-loss use).

import torch
from conftest import nntile_cpu

from torch_nntile.training import mse_loss


def reference_mse_loss(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    # The differentiable fallback used off device nntile.
    return float(scale) * (x * x).sum()


def test_mse_loss_backward_scales_by_grad_output():
    """``(w * mse_loss(x)).backward()`` must scale grads by ``w``."""
    torch.manual_seed(0)
    weight = 2.5

    x_cpu = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
    (weight * reference_mse_loss(x_cpu)).backward()
    grad_cpu = x_cpu.grad.detach().clone()

    x_nnt = x_cpu.detach().clone().to("nntile").requires_grad_(True)
    (weight * mse_loss(x_nnt, scale=1.0)).backward()

    assert torch.allclose(
        nntile_cpu(x_nnt.grad), grad_cpu, rtol=1e-4, atol=1e-4
    )


def test_mse_loss_backward_sum_of_weighted_losses():
    """Summing weighted losses must accumulate both upstream scales."""
    torch.manual_seed(1)
    w1, w2 = 0.5, 3.0

    x_cpu = torch.randn(3, 7, dtype=torch.float32, requires_grad=True)
    y_cpu = torch.randn(3, 7, dtype=torch.float32, requires_grad=True)
    (
        w1 * reference_mse_loss(x_cpu) + w2 * reference_mse_loss(y_cpu)
    ).backward()
    x_grad_cpu = x_cpu.grad.detach().clone()
    y_grad_cpu = y_cpu.grad.detach().clone()

    x_nnt = x_cpu.detach().clone().to("nntile").requires_grad_(True)
    y_nnt = y_cpu.detach().clone().to("nntile").requires_grad_(True)
    (w1 * mse_loss(x_nnt) + w2 * mse_loss(y_nnt)).backward()

    assert torch.allclose(
        nntile_cpu(x_nnt.grad), x_grad_cpu, rtol=1e-4, atol=1e-4
    )
    assert torch.allclose(
        nntile_cpu(y_nnt.grad), y_grad_cpu, rtol=1e-4, atol=1e-4
    )


def test_mse_loss_unit_grad_output_unchanged():
    """Plain ``loss.backward()`` keeps the historical unit-weight result."""
    torch.manual_seed(2)
    x_cpu = torch.randn(6, 2, dtype=torch.float32, requires_grad=True)
    reference_mse_loss(x_cpu).backward()
    grad_cpu = x_cpu.grad.detach().clone()

    x_nnt = x_cpu.detach().clone().to("nntile").requires_grad_(True)
    mse_loss(x_nnt).backward()

    assert torch.allclose(
        nntile_cpu(x_nnt.grad), grad_cpu, rtol=1e-4, atol=1e-4
    )
