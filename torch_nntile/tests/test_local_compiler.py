# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_local_compiler.py
# Public compile_graph() is off unless NNTILE_ENABLE_LOCAL_COMPILER=1.

from __future__ import annotations

import pytest
import torch
from conftest import nntile_cpu

import torch_nntile

_ENV = "NNTILE_ENABLE_LOCAL_COMPILER"


def test_compile_graph_raises_when_process_flag_is_off():
    """Default CI job leaves the env unset; skip if this process opted in."""
    if torch_nntile.local_compiler_enabled():
        pytest.skip(f"{_ENV}=1 in this process")
    with pytest.raises(RuntimeError, match="disabled by default"):
        torch_nntile.compile_graph()


@pytest.mark.parametrize("value", [None, "", "0", "true", "yes", "1 "])
def test_compile_graph_disabled_unless_one(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, value)
    assert torch_nntile.local_compiler_enabled() is False
    with pytest.raises(RuntimeError, match="disabled by default"):
        torch_nntile.compile_graph()


def test_compile_graph_with_flag_empty_pending(monkeypatch):
    monkeypatch.setenv(_ENV, "1")
    assert torch_nntile.local_compiler_enabled() is True
    torch_nntile.compile_graph()


def test_compile_graph_with_flag_compiles_pending_add(monkeypatch):
    monkeypatch.setenv(_ENV, "1")
    with torch.no_grad():
        lhs = torch.tensor([1.0, 2.0], dtype=torch.float32).to("nntile")
        rhs = torch.tensor([3.0, 4.0], dtype=torch.float32).to("nntile")
    out = lhs + rhs
    assert torch_nntile.has_pending_graph()
    torch_nntile.compile_graph()
    torch_nntile.run()
    result = nntile_cpu(out)
    torch.testing.assert_close(
        result,
        torch.tensor([4.0, 6.0], dtype=torch.float32),
    )
