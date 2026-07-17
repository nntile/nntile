# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_torch_native_add_parity.py
# Parity for torch-native StarPU add on device=nntile.

import torch
from conftest import nntile_cpu

import torch_nntile


def test_add_matches_cpu():
    a_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b_cpu = torch.tensor([[0.5, -1.0], [2.0, 0.25]])

    z = a_cpu.to("nntile") + b_cpu.to("nntile")

    assert z.device.type == "nntile"
    assert torch.allclose(nntile_cpu(z), a_cpu + b_cpu)


def test_add_with_alpha_matches_cpu():
    a_cpu = torch.tensor([1.0, -2.0, 3.5])
    b_cpu = torch.tensor([0.25, 4.0, -1.5])

    z = torch.add(a_cpu.to("nntile"), b_cpu.to("nntile"), alpha=2.0)
    expected = torch.add(a_cpu, b_cpu, alpha=2.0)
    assert torch.allclose(nntile_cpu(z), expected)


def test_add_2d_randn_parity():
    torch.manual_seed(0)
    shape = (4, 6)
    a_cpu = torch.randn(shape, dtype=torch.float32)
    b_cpu = torch.randn(shape, dtype=torch.float32)

    z_nntile = nntile_cpu(a_cpu.to("nntile") + b_cpu.to("nntile"))
    z_cpu = a_cpu + b_cpu
    assert torch.allclose(z_nntile, z_cpu, rtol=1e-5, atol=1e-5)
