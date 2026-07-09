# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_adam_step_parity.py
# Fused Adam / AdamW parity: CPU reference vs nntile tensor ops.

import pytest
import torch

import torch_nntile
from torch_nntile import _C
from conftest import nntile_cpu
from torch_nntile.training import (
    Adam, AdamW, _AdamBase, fused_adam_step, fused_adamw_step)

pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


def _reference_adam_values(
    param: torch.Tensor,
    grad: torch.Tensor,
    first_moment: torch.Tensor,
    second_moment: torch.Tensor,
    num_iter: int,
    beta_1: float,
    beta_2: float,
    eps: float,
    lr: float,
    weight_decay: float,
    *,
    decoupled_weight_decay: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _AdamBase._cpu_adam_values(
        param,
        grad,
        first_moment,
        second_moment,
        num_iter,
        beta_1,
        beta_2,
        eps,
        lr,
        weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
    )


@pytest.mark.parametrize(
    "shape,lr,beta_1,beta_2,eps,weight_decay",
    [
        ((6, 4), 1e-3, 0.9, 0.999, 1e-8, 0.0),
        ((3, 5), 5e-4, 0.9, 0.95, 1e-8, 1e-2),
        ((4,), 2e-3, 0.85, 0.99, 1e-6, 0.01),
    ],
)
def test_adam_step_matches_reference(
    shape, lr, beta_1, beta_2, eps, weight_decay
):
    torch.manual_seed(0)
    param_cpu = torch.randn(shape, dtype=torch.float32)
    grad_cpu = torch.randn(shape, dtype=torch.float32)
    m_cpu = torch.zeros(shape, dtype=torch.float32)
    v_cpu = torch.zeros(shape, dtype=torch.float32)

    expected_p, expected_m, expected_v = _reference_adam_values(
        param_cpu,
        grad_cpu,
        m_cpu,
        v_cpu,
        num_iter=1,
        beta_1=beta_1,
        beta_2=beta_2,
        eps=eps,
        lr=lr,
        weight_decay=weight_decay,
        decoupled_weight_decay=False,
    )

    param_nnt = param_cpu.clone().to("nntile")
    grad_nnt = grad_cpu.clone().to("nntile")
    m_nnt = m_cpu.clone().to("nntile")
    v_nnt = v_cpu.clone().to("nntile")

    _C.adam_step(
        param_nnt,
        grad_nnt,
        m_nnt,
        v_nnt,
        1,
        lr,
        beta_1,
        beta_2,
        eps,
        weight_decay,
    )

    assert torch.allclose(nntile_cpu(param_nnt), expected_p, rtol=1e-4, atol=1e-4)
    assert torch.allclose(nntile_cpu(m_nnt), expected_m, rtol=1e-4, atol=1e-4)
    assert torch.allclose(nntile_cpu(v_nnt), expected_v, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape,lr,beta_1,beta_2,eps,weight_decay",
    [
        ((6, 4), 1e-3, 0.9, 0.999, 1e-8, 0.0),
        ((3, 5), 5e-4, 0.9, 0.95, 1e-8, 1e-2),
        ((4,), 2e-3, 0.85, 0.99, 1e-6, 0.01),
    ],
)
def test_adamw_step_matches_reference(
    shape, lr, beta_1, beta_2, eps, weight_decay
):
    torch.manual_seed(1)
    param_cpu = torch.randn(shape, dtype=torch.float32)
    grad_cpu = torch.randn(shape, dtype=torch.float32)
    m_cpu = torch.zeros(shape, dtype=torch.float32)
    v_cpu = torch.zeros(shape, dtype=torch.float32)

    expected_p, expected_m, expected_v = _reference_adam_values(
        param_cpu,
        grad_cpu,
        m_cpu,
        v_cpu,
        num_iter=1,
        beta_1=beta_1,
        beta_2=beta_2,
        eps=eps,
        lr=lr,
        weight_decay=weight_decay,
        decoupled_weight_decay=True,
    )

    param_nnt = param_cpu.clone().to("nntile")
    grad_nnt = grad_cpu.clone().to("nntile")
    m_nnt = m_cpu.clone().to("nntile")
    v_nnt = v_cpu.clone().to("nntile")

    _C.adamw_step(
        param_nnt,
        grad_nnt,
        m_nnt,
        v_nnt,
        1,
        lr,
        beta_1,
        beta_2,
        eps,
        weight_decay,
    )

    assert torch.allclose(nntile_cpu(param_nnt), expected_p, rtol=1e-4, atol=1e-4)
    assert torch.allclose(nntile_cpu(m_nnt), expected_m, rtol=1e-4, atol=1e-4)
    assert torch.allclose(nntile_cpu(v_nnt), expected_v, rtol=1e-4, atol=1e-4)


def test_adam_optimizer_multistep():
    torch.manual_seed(2)
    shape = (5, 3)
    param_cpu = torch.randn(shape, dtype=torch.float32)
    param_nnt = param_cpu.clone().to("nntile")

    cpu_opt = Adam([param_cpu], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
    nnt_opt = Adam([param_nnt], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)

    for _ in range(3):
        grad = torch.randn(shape, dtype=torch.float32)
        param_cpu.grad = grad.clone()
        param_nnt.grad = grad.clone().to("nntile")
        cpu_opt.step()
        nnt_opt.step()
        torch_nntile.compile_graph()
        torch_nntile.run()

    assert torch.allclose(nntile_cpu(param_nnt), param_cpu, rtol=1e-4, atol=1e-4)


def test_adamw_optimizer_multistep():
    torch.manual_seed(3)
    shape = (4, 2)
    param_cpu = torch.randn(shape, dtype=torch.float32)
    param_nnt = param_cpu.clone().to("nntile")

    cpu_opt = AdamW([param_cpu], lr=2e-3, betas=(0.9, 0.999), weight_decay=0.1)
    nnt_opt = AdamW([param_nnt], lr=2e-3, betas=(0.9, 0.999), weight_decay=0.1)

    for _ in range(4):
        grad = torch.randn(shape, dtype=torch.float32)
        param_cpu.grad = grad.clone()
        param_nnt.grad = grad.clone().to("nntile")
        cpu_opt.step()
        nnt_opt.step()
        torch_nntile.compile_graph()
        torch_nntile.run()

    assert torch.allclose(nntile_cpu(param_nnt), param_cpu, rtol=1e-4, atol=1e-4)


def test_fused_adam_step_plain():
    torch.manual_seed(4)
    shape = (3, 2)
    param = torch.randn(shape, dtype=torch.float32)
    grad = torch.randn(shape, dtype=torch.float32)
    m = torch.zeros(shape, dtype=torch.float32)
    v = torch.zeros(shape, dtype=torch.float32)
    expected_p, _, _ = _reference_adam_values(
        param,
        grad,
        m,
        v,
        num_iter=1,
        beta_1=0.9,
        beta_2=0.999,
        eps=1e-8,
        lr=1e-3,
        weight_decay=0.0,
        decoupled_weight_decay=False,
    )

    param_nnt = param.clone().to("nntile")
    param_nnt.grad = grad.clone().to("nntile")
    fused_adam_step([param_nnt], learning_rate=1e-3)

    assert torch.allclose(nntile_cpu(param_nnt), expected_p, rtol=1e-4, atol=1e-4)


def test_fused_adamw_step_plain():
    torch.manual_seed(5)
    shape = (3, 2)
    param = torch.randn(shape, dtype=torch.float32)
    grad = torch.randn(shape, dtype=torch.float32)
    m = torch.zeros(shape, dtype=torch.float32)
    v = torch.zeros(shape, dtype=torch.float32)
    expected_p, _, _ = _reference_adam_values(
        param,
        grad,
        m,
        v,
        num_iter=1,
        beta_1=0.9,
        beta_2=0.999,
        eps=1e-8,
        lr=1e-3,
        weight_decay=0.01,
        decoupled_weight_decay=True,
    )

    param_nnt = param.clone().to("nntile")
    param_nnt.grad = grad.clone().to("nntile")
    fused_adamw_step([param_nnt], learning_rate=1e-3)

    assert torch.allclose(nntile_cpu(param_nnt), expected_p, rtol=1e-4, atol=1e-4)


def test_adamw_default_weight_decay():
    opt = AdamW([torch.randn(2)], lr=1e-3)
    assert opt.param_groups[0]["weight_decay"] == 0.01


def test_adam_step_accepts_keyword_lr():
    torch.manual_seed(6)
    shape = (2, 2)
    param = torch.randn(shape, dtype=torch.float32).to("nntile")
    grad = torch.randn(shape, dtype=torch.float32).to("nntile")
    m = torch.empty(shape, dtype=torch.float32, device="nntile")
    v = torch.empty(shape, dtype=torch.float32, device="nntile")

    _C.adam_step(
        param,
        grad,
        m,
        v,
        num_iter=1,
        lr=1e-3,
    )

    assert torch.isfinite(nntile_cpu(param)).all()
