# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_add_parity.py
# Parity tests for nntile add via TensorGraph (libnntile).

import torch
from conftest import nntile_cpu

import torch_nntile


def test_add_matches_cpu():
    a_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b_cpu = torch.tensor([[0.5, -1.0], [2.0, 0.25]])

    a = a_cpu.to("nntile")
    b = b_cpu.to("nntile")
    z = a + b

    assert z.device.type == "nntile"
    assert torch.allclose(nntile_cpu(z), a_cpu + b_cpu)


def test_add_with_alpha_matches_cpu():
    a_cpu = torch.tensor([1.0, -2.0, 3.5])
    b_cpu = torch.tensor([0.25, 4.0, -1.5])

    a = a_cpu.to("nntile")
    b = b_cpu.to("nntile")
    z = torch.add(a, b, alpha=2.0)

    expected = torch.add(a_cpu, b_cpu, alpha=2.0)
    assert torch.allclose(nntile_cpu(z), expected)


def test_add_2d_shape_parity():
    shape = (4, 6)
    a_cpu = torch.randn(shape, dtype=torch.float32)
    b_cpu = torch.randn(shape, dtype=torch.float32)

    z_nntile = nntile_cpu(a_cpu.to("nntile") + b_cpu.to("nntile"))
    z_cpu = a_cpu + b_cpu

    assert torch.allclose(z_nntile, z_cpu, rtol=1e-5, atol=1e-5)


def test_add_inplace_broadcast_matches_cpu():
    a_cpu = torch.randn(2, 3, 4, dtype=torch.float32)
    b_cpu = torch.randn(1, 3, 4, dtype=torch.float32)
    a_nnt = a_cpu.clone().to("nntile")
    b_nnt = b_cpu.to("nntile")
    a_cpu.add_(b_cpu)
    a_nnt.add_(b_nnt)
    torch.testing.assert_close(nntile_cpu(a_nnt), a_cpu, rtol=1e-5, atol=1e-5)


def test_add_out_of_place_broadcast_matches_cpu():
    a_cpu = torch.randn(2, 3, 4, dtype=torch.float32)
    b_cpu = torch.randn(1, 3, 4, dtype=torch.float32)
    ref = a_cpu + b_cpu
    got = a_cpu.to("nntile") + b_cpu.to("nntile")
    torch.testing.assert_close(nntile_cpu(got), ref, rtol=1e-5, atol=1e-5)
