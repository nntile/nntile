# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_add_inplace_parity.py
# Parity tests for nntile add_ via TensorGraph (libnntile).

import torch
import pytest

import torch_nntile
from torch_nntile import _C
from conftest import nntile_cpu


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


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
