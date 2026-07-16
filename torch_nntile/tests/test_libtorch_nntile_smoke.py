# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_libtorch_nntile_smoke.py
# Minimal libtorch_nntile / device=nntile smoke (CI + local).

from __future__ import annotations

import torch
import pytest

import torch_nntile
from torch_nntile import _C
from conftest import nntile_cpu


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


def test_libtorch_nntile_smoke_add():
    """Import + tiny add on nntile (same idea as tools/smoke_test_wheel.py)."""
    assert _C.has_libnntile()
    assert torch_nntile.is_context_initialized()

    with torch.no_grad():
        lhs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).to("nntile")
        rhs = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32).to("nntile")
    out = lhs + rhs
    torch_nntile.compile_graph()
    torch_nntile.run()
    result = nntile_cpu(out)
    torch.testing.assert_close(
        result,
        torch.tensor([5.0, 7.0, 9.0], dtype=torch.float32),
    )
