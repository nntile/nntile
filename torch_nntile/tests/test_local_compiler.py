# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_local_compiler.py
# Local TensorGraph lower is off unless NNTILE_ENABLE_LOCAL_COMPILER=1.
# Covers public compile_graph(), legacy execute(), and .to("cpu") auto-flush.

from __future__ import annotations

import pytest
import torch
from conftest import nntile_cpu

import torch_nntile

_ENV = "NNTILE_ENABLE_LOCAL_COMPILER"
_DISABLED = [None, "", "0", "true", "yes", "1 "]


def _set_flag(monkeypatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, value)


def test_compile_graph_raises_when_process_flag_is_off():
    """Default CI job leaves the env unset; skip if this process opted in."""
    if torch_nntile.local_compiler_enabled():
        pytest.skip(f"{_ENV}=1 in this process")
    with pytest.raises(RuntimeError, match="disabled by default"):
        torch_nntile.compile_graph()


@pytest.mark.parametrize("value", _DISABLED)
def test_compile_graph_disabled_unless_one(monkeypatch, value):
    _set_flag(monkeypatch, value)
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


def test_execute_raises_when_process_flag_is_off():
    if torch_nntile.local_compiler_enabled():
        pytest.skip(f"{_ENV}=1 in this process")
    with pytest.raises(RuntimeError, match="disabled by default"):
        torch_nntile.execute()


@pytest.mark.parametrize("value", _DISABLED)
def test_execute_disabled_unless_one(monkeypatch, value):
    _set_flag(monkeypatch, value)
    assert torch_nntile.local_compiler_enabled() is False
    with pytest.raises(RuntimeError, match="disabled by default"):
        torch_nntile.execute()
    with pytest.raises(RuntimeError, match="disabled by default"):
        torch_nntile._C.execute()


def test_execute_with_flag_runs_pending_add(monkeypatch):
    monkeypatch.setenv(_ENV, "1")
    with torch.no_grad():
        lhs = torch.tensor([1.0, 2.0], dtype=torch.float32).to("nntile")
        rhs = torch.tensor([3.0, 4.0], dtype=torch.float32).to("nntile")
    out = lhs + rhs
    assert torch_nntile.has_pending_graph()
    torch_nntile.execute()
    result = nntile_cpu(out)
    torch.testing.assert_close(
        result,
        torch.tensor([4.0, 6.0], dtype=torch.float32),
    )


def _pending_add():
    with torch.no_grad():
        lhs = torch.tensor([1.0, 2.0], dtype=torch.float32).to("nntile")
        rhs = torch.tensor([3.0, 4.0], dtype=torch.float32).to("nntile")
    return lhs + rhs


@pytest.mark.parametrize("value", _DISABLED)
def test_to_cpu_does_not_lower_pending_add(monkeypatch, value):
    _set_flag(monkeypatch, value)
    out = _pending_add()
    try:
        assert torch_nntile.has_pending_graph()
        with pytest.raises(RuntimeError, match="disabled by default"):
            out.to("cpu")
        assert torch_nntile.has_pending_graph()
    finally:
        torch_nntile.reset_graph_session()


def test_to_cpu_with_flag_auto_flushes_pending_add(monkeypatch):
    monkeypatch.setenv(_ENV, "1")
    out = _pending_add()
    assert torch_nntile.has_pending_graph()
    result = out.to("cpu")
    assert not torch_nntile.has_pending_graph()
    torch.testing.assert_close(
        result,
        torch.tensor([4.0, 6.0], dtype=torch.float32),
    )
