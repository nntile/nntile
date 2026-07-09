# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_bmm.py

import pytest
import torch

pytest.importorskip("torch_nntile")
import torch_nntile  # noqa: E402
from conftest import nntile_cpu


def test_bmm_parity():
    torch.manual_seed(0)
    a_cpu = torch.randn(4, 3, 5)
    b_cpu = torch.randn(4, 5, 2)
    a_nnt = a_cpu.to("nntile")
    b_nnt = b_cpu.to("nntile")
    out_cpu = torch.bmm(a_cpu, b_cpu)
    out_nnt = nntile_cpu(torch.bmm(a_nnt, b_nnt))
    torch.testing.assert_close(out_nnt, out_cpu, rtol=1e-5, atol=1e-5)


def test_matmul_3d_parity():
    torch.manual_seed(1)
    a_cpu = torch.randn(2, 3, 4)
    b_cpu = torch.randn(2, 4, 5)
    a_nnt = a_cpu.to("nntile")
    b_nnt = b_cpu.to("nntile")
    out_cpu = torch.matmul(a_cpu, b_cpu)
    out_nnt = nntile_cpu(torch.matmul(a_nnt, b_nnt))
    torch.testing.assert_close(out_nnt, out_cpu, rtol=1e-5, atol=1e-5)
