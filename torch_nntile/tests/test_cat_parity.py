# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_cat_parity.py
# Parity tests for nntile torch.cat via TensorGraph (libnntile).

import subprocess
import sys
import textwrap
from pathlib import Path

import torch
import pytest

from torch_nntile import _C
from conftest import nntile_cpu, subprocess_environ


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


def _run_graph_subprocess(script: str) -> None:
    env = subprocess_environ()
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"subprocess failed ({proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def test_cat_1d_dim0():
    a_cpu = torch.tensor([1.0, 2.0, 3.0])
    b_cpu = torch.tensor([4.0, 5.0])
    expected = torch.cat([a_cpu, b_cpu], dim=0)

    result = torch.cat([a_cpu.to("nntile"), b_cpu.to("nntile")], dim=0)
    assert result.device.type == "nntile"
    assert torch.allclose(nntile_cpu(result), expected)


def test_cat_2d_dim1():
    a_cpu = torch.randn(2, 4, dtype=torch.float32)
    b_cpu = torch.randn(2, 5, dtype=torch.float32)
    expected = torch.cat([a_cpu, b_cpu], dim=1)

    result = torch.cat(
        [a_cpu.to("nntile"), b_cpu.to("nntile")],
        dim=1,
    )
    assert torch.allclose(nntile_cpu(result), expected, rtol=1e-5, atol=1e-5)


def test_cat_2d_dim0():
    a_cpu = torch.randn(3, 4, dtype=torch.float32)
    b_cpu = torch.randn(2, 4, dtype=torch.float32)
    expected = torch.cat([a_cpu, b_cpu], dim=0)

    result = torch.cat(
        [a_cpu.to("nntile"), b_cpu.to("nntile")],
        dim=0,
    )
    assert torch.allclose(nntile_cpu(result), expected, rtol=1e-5, atol=1e-5)


def test_cat_3d():
    a_cpu = torch.randn(2, 2, 2, dtype=torch.float32)
    b_cpu = torch.randn(2, 2, 3, dtype=torch.float32)
    expected = torch.cat([a_cpu, b_cpu], dim=2)

    result = torch.cat(
        [a_cpu.to("nntile"), b_cpu.to("nntile")],
        dim=2,
    )
    assert torch.allclose(nntile_cpu(result), expected, rtol=1e-5, atol=1e-5)


def test_cat_negative_dim():
    a_cpu = torch.randn(2, 3, dtype=torch.float32)
    b_cpu = torch.randn(2, 4, dtype=torch.float32)
    expected = torch.cat([a_cpu, b_cpu], dim=-1)

    result = torch.cat(
        [a_cpu.to("nntile"), b_cpu.to("nntile")],
        dim=-1,
    )
    assert torch.allclose(nntile_cpu(result), expected, rtol=1e-5, atol=1e-5)


def test_cat_many_tensors():
    tensors_cpu = [torch.randn(2, 3, dtype=torch.float32) for _ in range(4)]
    expected = torch.cat(tensors_cpu, dim=1)

    tensors_nntile = [t.to("nntile") for t in tensors_cpu]
    result = torch.cat(tensors_nntile, dim=1)
    assert torch.allclose(nntile_cpu(result), expected, rtol=1e-5, atol=1e-5)


def test_cat_single_tensor_is_noop():
    a_cpu = torch.randn(2, 3, dtype=torch.float32)
    a = a_cpu.to("nntile")
    result = torch.cat([a], dim=0)
    assert result is a
    assert torch.allclose(nntile_cpu(result), a_cpu, rtol=1e-5, atol=1e-5)


def test_cat_out_variant():
    a_cpu = torch.randn(2, 3, dtype=torch.float32)
    b_cpu = torch.randn(2, 4, dtype=torch.float32)
    expected = torch.cat([a_cpu, b_cpu], dim=1)

    out = torch.empty(2, 7, device="nntile")
    torch.cat([a_cpu.to("nntile"), b_cpu.to("nntile")], dim=1, out=out)
    assert torch.allclose(nntile_cpu(out), expected, rtol=1e-5, atol=1e-5)


def test_cat_deferred_until_compile():
    _run_graph_subprocess(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1,
            ncuda=0,
            cpu_fallback=False,
        )
        torch_nntile.restrict_cpu()
        a_cpu = torch.randn(2, 3, dtype=torch.float32)
        b_cpu = torch.randn(2, 4, dtype=torch.float32)
        expected = torch.cat([a_cpu, b_cpu], dim=1)

        result = torch.cat(
            [a_cpu.to("nntile"), b_cpu.to("nntile")],
            dim=1,
        )
        assert torch_nntile.has_pending_graph()
        torch_nntile.compile_graph()
        torch_nntile.run()
        assert torch.allclose(result.cpu(), expected, rtol=1e-5, atol=1e-5)
        """
    )
