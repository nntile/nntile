# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_addmm_parity.py

import pytest
import torch
from conftest import nntile_cpu

pytest.importorskip("torch_nntile")


def test_addmm_parity():
    torch.manual_seed(5)
    bias_cpu = torch.randn(4, 5)
    a_cpu = torch.randn(4, 3)
    b_cpu = torch.randn(3, 5)
    bias_nnt = bias_cpu.to("nntile")
    a_nnt = a_cpu.to("nntile")
    b_nnt = b_cpu.to("nntile")
    out_cpu = torch.addmm(bias_cpu, a_cpu, b_cpu, beta=0.5, alpha=2.0)
    out_nnt = nntile_cpu(
        torch.addmm(bias_nnt, a_nnt, b_nnt, beta=0.5, alpha=2.0)
    )
    torch.testing.assert_close(out_nnt, out_cpu, rtol=1e-5, atol=1e-5)
