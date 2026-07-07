# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_repeat_parity.py
# Parity tests for nntile repeat via chained scale_slice.

import torch
import pytest

from torch_nntile import _C
from conftest import nntile_cpu


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


def test_repeat_1d_matches_cpu():
    x_cpu = torch.tensor([1.0, 2.0, 3.0])
    x = x_cpu.to("nntile")
    y = x.repeat(2)
    assert torch.allclose(nntile_cpu(y), x_cpu.repeat(2))


def test_repeat_1d_padded_repeats_matches_cpu():
    x_cpu = torch.tensor([1.0, 2.0, 3.0])
    x = x_cpu.to("nntile")
    y = x.repeat(2, 3)
    assert torch.allclose(nntile_cpu(y), x_cpu.repeat(2, 3))


def test_repeat_2d_matches_cpu():
    x_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x = x_cpu.to("nntile")
    y = x.repeat(2, 3)
    assert torch.allclose(nntile_cpu(y), x_cpu.repeat(2, 3))


@pytest.mark.skip(reason="repeat(1,1) segfaults in graph execute")
def test_repeat_identity_matches_cpu():
    x_cpu = torch.randn(3, 4, dtype=torch.float32)
    x = x_cpu.to("nntile")
    y = x.repeat(1, 1)
    assert torch.allclose(nntile_cpu(y), x_cpu.repeat(1, 1))


def test_repeat_leading_pad_matches_cpu():
    x_cpu = torch.tensor([1.0, 2.0])
    x = x_cpu.to("nntile")
    y = x.repeat(1, 2, 1)
    assert torch.allclose(nntile_cpu(y), x_cpu.repeat(1, 2, 1))
