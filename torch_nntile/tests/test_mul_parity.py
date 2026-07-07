# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_mul_parity.py
# Parity tests for nntile mul via TensorGraph (libnntile).

import torch
import pytest

import torch_nntile
from torch_nntile import _C
from conftest import nntile_cpu


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


def test_mul_matches_cpu():
    a_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b_cpu = torch.tensor([[0.5, -1.0], [2.0, 0.25]])

    a = a_cpu.to("nntile")
    b = b_cpu.to("nntile")
    z = a * b

    assert z.device.type == "nntile"
    assert torch.allclose(nntile_cpu(z), a_cpu * b_cpu)


def test_mul_inplace_matches_cpu():
    a_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b_cpu = torch.tensor([[0.5, -1.0], [2.0, 0.25]])

    a = a_cpu.clone().to("nntile")
    b = b_cpu.to("nntile")
    expected = a_cpu * b_cpu

    a.mul_(b)
    assert torch.allclose(nntile_cpu(a), expected)


def test_mul_2d_shape_parity():
    shape = (4, 6)
    a_cpu = torch.randn(shape, dtype=torch.float32)
    b_cpu = torch.randn(shape, dtype=torch.float32)

    z_nntile = nntile_cpu(a_cpu.to("nntile") * b_cpu.to("nntile"))
    z_cpu = a_cpu * b_cpu

    assert torch.allclose(z_nntile, z_cpu, rtol=1e-5, atol=1e-5)
