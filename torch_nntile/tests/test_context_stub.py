# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_context_stub.py
# Context APIs without libnntile should fail clearly.

import pytest

import torch_nntile
from torch_nntile import _C


@pytest.mark.skipif(
    _C.has_libnntile(),
    reason="only for stub builds without libnntile",
)
def test_context_apis_require_libnntile():
    with pytest.raises(RuntimeError, match="libnntile"):
        torch_nntile.restrict_cuda()
    with pytest.raises(RuntimeError, match="libnntile"):
        torch_nntile.restore_where()
