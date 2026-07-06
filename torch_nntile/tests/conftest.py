# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/conftest.py
# Session-wide StarPU / nntile context for libnntile parity tests.

import sys
from pathlib import Path

# Layout is torch_nntile/torch_nntile/; when pytest runs from the repo root,
# cwd shadows the editable install unless the project root is on sys.path.
_pkg_root = Path(__file__).resolve().parent.parent
if _pkg_root.name == "torch_nntile":
    _root = str(_pkg_root)
    if _root not in sys.path:
        sys.path.insert(0, _root)

import pytest
import torch

import torch_nntile
from torch_nntile import _C


def pytest_sessionstart(session) -> None:
    """Configure nntile before collection or any libnntile-backed op."""
    del session
    if not _C.has_libnntile():
        return
    if not torch_nntile.is_context_initialized():
        try:
            torch_nntile.init_context(
                ncpu=1,
                ncuda=0,
                verbose=0,
                cpu_fallback=False,
            )
        except RuntimeError:
            # Configuration was locked earlier in this process.
            pass
    torch_nntile.restrict_cpu()


@pytest.fixture(autouse=True)
def _reset_nntile_graph_session_after_test():
    """Isolate parity tests: stale TensorGraph sessions corrupt later tests."""
    yield
    if _C.has_libnntile():
        torch_nntile.reset_graph_session()


def nntile_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Copy an nntile tensor to CPU, flushing a pending TensorGraph first."""
    if (
        _C.has_libnntile()
        and tensor.device.type == "nntile"
        and torch_nntile.has_pending_graph()
    ):
        torch_nntile.compile_graph()
        torch_nntile.run()
    return tensor.cpu()
