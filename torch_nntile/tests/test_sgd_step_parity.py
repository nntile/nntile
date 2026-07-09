# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_sgd_step_parity.py
# Fused SGD parity: CPU PyTorch vs nntile tensor::sgd_step.

import torch
import pytest

import torch_nntile
from torch_nntile import _C
from torch_nntile.training import SGD, fused_sgd_step
from conftest import nntile_cpu


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)

def _reference_cpu_sgd(
    param: torch.Tensor,
    grad: torch.Tensor,
    velocity: torch.Tensor,
    num_iter: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    dampening: float,
    nesterov: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    p = param.detach().clone()
    v = velocity.detach().clone()
    g = grad.detach().clone()
    if weight_decay != 0:
        g = g + weight_decay * p
    if momentum != 0:
        if num_iter == 1:
            v = g.clone()
        else:
            v = momentum * v + (1 - dampening) * g
        if nesterov:
            g = g + momentum * v
        else:
            g = v
    p = p - lr * g
    return p, v


@pytest.mark.parametrize(
    "shape,lr,momentum,weight_decay,dampening,nesterov",
    [
        ((6, 4), 0.1, 0.0, 0.0, 0.0, False),
        ((3, 5), 0.05, 0.9, 0.0, 0.0, False),
        ((2, 3), 0.01, 0.9, 0.01, 0.0, False),
        ((4,), 0.2, 0.8, 0.0, 0.0, True),
    ],
)
def test_sgd_step_matches_reference(
    shape, lr, momentum, weight_decay, dampening, nesterov
):
    torch.manual_seed(0)
    param_cpu = torch.randn(shape, dtype=torch.float32)
    grad_cpu = torch.randn(shape, dtype=torch.float32)
    velocity_cpu = torch.zeros(shape, dtype=torch.float32)

    expected_p, expected_v = _reference_cpu_sgd(
        param_cpu,
        grad_cpu,
        velocity_cpu,
        num_iter=1,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        dampening=dampening,
        nesterov=nesterov,
    )

    param_nnt = param_cpu.clone().to("nntile")
    grad_nnt = grad_cpu.clone().to("nntile")
    velocity_nnt = velocity_cpu.clone().to("nntile")

    _C.sgd_step(
        param_nnt,
        grad_nnt,
        velocity_nnt,
        1,
        lr,
        momentum,
        weight_decay,
        dampening,
        nesterov,
    )

    assert torch.allclose(nntile_cpu(param_nnt), expected_p, rtol=1e-4, atol=1e-4)
    assert torch.allclose(
        nntile_cpu(velocity_nnt), expected_v, rtol=1e-4, atol=1e-4
    )


def test_sgd_optimizer_matches_torch():
    torch.manual_seed(1)
    shape = (8, 3)
    param_cpu = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    param_nnt = param_cpu.detach().clone().to("nntile").requires_grad_(True)

    grad = torch.randn(shape, dtype=torch.float32)
    param_cpu.grad = grad.clone()
    param_nnt.grad = grad.clone().to("nntile")

    torch_opt = torch.optim.SGD([param_cpu], lr=0.1, momentum=0.9)
    nnt_opt = SGD([param_nnt], lr=0.1, momentum=0.9)

    torch_opt.step()
    nnt_opt.step()

    assert torch.allclose(nntile_cpu(param_nnt), param_cpu.detach(), rtol=1e-4, atol=1e-4)


def test_sgd_optimizer_multistep_momentum():
    torch.manual_seed(2)
    shape = (5,)
    param_cpu = torch.randn(shape, dtype=torch.float32)
    param_nnt = param_cpu.clone().to("nntile")

    torch_opt = torch.optim.SGD([param_cpu], lr=0.2, momentum=0.9)
    nnt_opt = SGD([param_nnt], lr=0.2, momentum=0.9)

    for step in range(3):
        grad = torch.randn(shape, dtype=torch.float32)
        param_cpu.grad = grad.clone()
        param_nnt.grad = grad.clone().to("nntile")
        torch_opt.step()
        nnt_opt.step()
        torch_nntile.compile_graph()
        torch_nntile.run()

    assert torch.allclose(nntile_cpu(param_nnt), param_cpu, rtol=1e-4, atol=1e-4)


def test_fused_sgd_step_plain():
    torch.manual_seed(3)
    param = torch.randn(4, 2, dtype=torch.float32)
    grad = torch.randn(4, 2, dtype=torch.float32)
    expected = param - 0.1 * grad

    param_nnt = param.clone().to("nntile")
    param_nnt.grad = grad.clone().to("nntile")
    fused_sgd_step([param_nnt], learning_rate=0.1)

    assert torch.allclose(nntile_cpu(param_nnt), expected, rtol=1e-4, atol=1e-4)
