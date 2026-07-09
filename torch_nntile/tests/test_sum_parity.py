# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_sum_parity.py
# Parity tests for nntile sum via TensorGraph (libnntile).

import torch
import pytest

import torch_nntile
from torch_nntile import _C
from conftest import nntile_cpu


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


def test_sum_multi_axis_keepdim_matches_cpu():
    x_cpu = torch.randn(2, 3, 4, dtype=torch.float32)
    x_nnt = x_cpu.to("nntile")
    ref = x_cpu.sum(dim=(0, 1), keepdim=True)
    got = x_nnt.sum(dim=(0, 1), keepdim=True)
    torch.testing.assert_close(nntile_cpu(got), ref, rtol=1e-4, atol=1e-4)


def test_sum_empty_dim_matches_cpu():
    x_cpu = torch.randn(2, 3, 4, dtype=torch.float32)
    x_nnt = x_cpu.to("nntile")
    ref = x_cpu.sum(dim=())
    got = x_nnt.sum(dim=())
    torch.testing.assert_close(nntile_cpu(got), ref, rtol=1e-4, atol=1e-4)
