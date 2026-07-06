# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_hypot_parity.py
# Parity tests for nntile hypot via TensorGraph (libnntile).

import torch
import pytest

import torch_nntile
from torch_nntile import _C
from conftest import nntile_cpu


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


def test_hypot_matches_cpu():
    a_cpu = torch.tensor([[3.0, 5.0], [8.0, 6.0]])
    b_cpu = torch.tensor([[4.0, 12.0], [15.0, 8.0]])

    a = a_cpu.to("nntile")
    b = b_cpu.to("nntile")
    z = torch.hypot(a, b)

    assert z.device.type == "nntile"
    assert torch.allclose(nntile_cpu(z), torch.hypot(a_cpu, b_cpu))


def test_hypot_out_matches_cpu():
    a_cpu = torch.tensor([[3.0, 5.0], [8.0, 6.0]])
    b_cpu = torch.tensor([[4.0, 12.0], [15.0, 8.0]])

    a = a_cpu.to("nntile")
    b = b_cpu.to("nntile")
    out = torch.empty_like(a, device="nntile")
    torch.hypot(a, b, out=out)

    assert torch.allclose(nntile_cpu(out), torch.hypot(a_cpu, b_cpu))


def test_hypot_2d_shape_parity():
    shape = (4, 6)
    a_cpu = torch.randn(shape, dtype=torch.float32)
    b_cpu = torch.randn(shape, dtype=torch.float32)

    z_nntile = torch.hypot(a_cpu.to("nntile"), b_cpu.to("nntile")).cpu()
    z_cpu = torch.hypot(a_cpu, b_cpu)

    assert torch.allclose(z_nntile, z_cpu, rtol=1e-5, atol=1e-5)
