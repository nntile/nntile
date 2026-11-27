# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_add_inplace_parity.py
# Parity tests for nntile add_ via TensorGraph (libnntile).

import torch
from conftest import nntile_cpu

import torch_nntile


def test_add_inplace_matches_cpu():
    a_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b_cpu = torch.tensor([[0.5, -1.0], [2.0, 0.25]])

    a = a_cpu.clone().to("nntile")
    b = b_cpu.to("nntile")
    expected = a_cpu + b_cpu

    a.add_(b)
    assert torch.allclose(nntile_cpu(a), expected)


def test_add_inplace_with_alpha_matches_cpu():
    a_cpu = torch.tensor([1.0, -2.0, 3.5])
    b_cpu = torch.tensor([0.25, 4.0, -1.5])

    a = a_cpu.clone().to("nntile")
    b = b_cpu.to("nntile")
    expected = torch.add(a_cpu, b_cpu, alpha=2.0)

    a.add_(b, alpha=2.0)
    assert torch.allclose(nntile_cpu(a), expected)


def test_add_inplace_2d_shape_parity():
    shape = (4, 6)
    a_cpu = torch.randn(shape, dtype=torch.float32)
    b_cpu = torch.randn(shape, dtype=torch.float32)

    a_nntile = a_cpu.clone().to("nntile")
    b_nntile = b_cpu.to("nntile")
    a_nntile.add_(b_nntile)

    a_cpu.add_(b_cpu)
    assert torch.allclose(nntile_cpu(a_nntile), a_cpu, rtol=1e-5, atol=1e-5)


def test_add_inplace_rebinds_across_steps():
    """SSA add_ must rebind TensorRef so later steps see updated weights.

    Stock ``torch.optim.SGD`` relies on this for ``param.add_(grad, alpha=-lr)``.
    """
    a_cpu = torch.tensor([1.0, 2.0, 3.0])
    b_cpu = torch.tensor([0.5, -1.0, 0.25])
    a = a_cpu.clone().to("nntile")
    b = b_cpu.to("nntile")

    a.add_(b)
    expected = a_cpu + b_cpu
    assert torch.allclose(nntile_cpu(a), expected)

    a.add_(b, alpha=-0.5)
    expected = expected + (-0.5) * b_cpu
    assert torch.allclose(nntile_cpu(a), expected)

    # Simulate default SGD (momentum=0): p.add_(g, alpha=-lr)
    p_cpu = torch.tensor([1.0, 1.0])
    g_cpu = torch.tensor([0.2, -0.4])
    p = p_cpu.clone().to("nntile")
    g = g_cpu.to("nntile")
    lr = 0.5
    with torch.no_grad():
        p.add_(g, alpha=-lr)
        p.add_(g, alpha=-lr)
    expected_p = p_cpu + 2 * ((-lr) * g_cpu)
    assert torch.allclose(nntile_cpu(p), expected_p)
