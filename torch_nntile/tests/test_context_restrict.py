# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_context_restrict.py
# Tests for restrict_cuda / restore_where context controls.

import pytest

import torch_nntile
from torch_nntile import _C


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


def test_init_context_before_ops():
    if not torch_nntile.is_context_initialized():
        torch_nntile.init_context(ncpu=1, ncuda=0, verbose=0)
    torch_nntile.restrict_cpu()
    assert torch_nntile.is_context_initialized()
    torch_nntile.restore_where()


def test_restrict_restore_roundtrip():
    torch_nntile.restrict_cpu()
    torch_nntile.restore_where()
    torch_nntile.restrict_cuda()
    torch_nntile.restore_where()


def test_init_context_after_first_op_raises():
    import torch

    torch_nntile.restrict_cpu()
    with pytest.raises(RuntimeError, match="init_context"):
        torch_nntile.init_context(ncpu=2, ncuda=0)
