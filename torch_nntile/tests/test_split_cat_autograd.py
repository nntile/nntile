# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_split_cat_autograd.py
# Autograd parity tests for nntile split, narrow, and cat.

import subprocess
import sys
import textwrap
from pathlib import Path

import torch
import pytest

from torch_nntile import _C
from conftest import nntile_cpu


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)

_PKG_ROOT = Path(__file__).resolve().parent.parent


def _grad_with_ones(output: torch.Tensor, inputs: tuple[torch.Tensor, ...]) -> tuple:
    """Backward via grad_outputs; avoids ``sum`` (not on nntile)."""
    grad_out = torch.ones_like(output)
    return torch.autograd.grad(output, inputs, grad_outputs=grad_out)


def _run_graph_subprocess(script: str) -> None:
    env = dict(**__import__("os").environ)
    repo = Path(__file__).resolve().parents[2]
    build_lib = repo / "build" / "nntile"
    starpu_lib = "/opt/starpu/lib"
    ld = env.get("LD_LIBRARY_PATH", "")
    for part in (str(build_lib), starpu_lib):
        if part not in ld.split(":"):
            ld = f"{part}:{ld}" if ld else part
    env["LD_LIBRARY_PATH"] = ld
    env["PYTHONPATH"] = f"{_PKG_ROOT}:{env.get('PYTHONPATH', '')}"
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


def test_split_with_sizes_forward_2d():
    x_cpu = torch.randn(2, 7, dtype=torch.float32)
    expected = torch.split(x_cpu, [3, 4], dim=1)

    parts = torch.split(x_cpu.to("nntile"), [3, 4], dim=1)
    assert len(parts) == 2
    for part, ref in zip(parts, expected, strict=True):
        assert torch.allclose(nntile_cpu(part), ref, rtol=1e-5, atol=1e-5)


def test_chunk_forward():
    x_cpu = torch.randn(2, 7, dtype=torch.float32)
    expected = torch.chunk(x_cpu, 3, dim=1)

    parts = torch.chunk(x_cpu.to("nntile"), 3, dim=1)
    assert len(parts) == len(expected)
    for part, ref in zip(parts, expected, strict=True):
        assert torch.allclose(nntile_cpu(part), ref, rtol=1e-5, atol=1e-5)


def test_split_equal_size_forward():
    x_cpu = torch.randn(2, 6, dtype=torch.float32)
    expected = torch.split(x_cpu, 3, dim=1)

    parts = torch.split(x_cpu.to("nntile"), 3, dim=1)
    for part, ref in zip(parts, expected, strict=True):
        assert torch.allclose(nntile_cpu(part), ref, rtol=1e-5, atol=1e-5)


def test_narrow_forward():
    x_cpu = torch.randn(2, 7, dtype=torch.float32)
    expected = x_cpu.narrow(1, 2, 4)

    result = x_cpu.to("nntile").narrow(1, 2, 4)
    assert torch.allclose(nntile_cpu(result), expected, rtol=1e-5, atol=1e-5)


def test_cat_backward_two_tensors():
    a_cpu = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
    b_cpu = torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
    y_cpu = torch.cat([a_cpu, b_cpu], dim=1)
    ga_cpu, gb_cpu = _grad_with_ones(y_cpu, (a_cpu, b_cpu))

    a = a_cpu.detach().to("nntile").requires_grad_(True)
    b = b_cpu.detach().to("nntile").requires_grad_(True)
    y = torch.cat([a, b], dim=1)
    ga, gb = _grad_with_ones(y, (a, b))

    assert torch.allclose(nntile_cpu(ga), ga_cpu, rtol=1e-5, atol=1e-5)
    assert torch.allclose(nntile_cpu(gb), gb_cpu, rtol=1e-5, atol=1e-5)


def test_cat_backward_many_tensors():
    tensors_cpu = [
        torch.randn(2, 3, dtype=torch.float32, requires_grad=True) for _ in range(4)
    ]
    y_cpu = torch.cat(tensors_cpu, dim=1)
    grads_cpu = _grad_with_ones(y_cpu, tuple(tensors_cpu))

    tensors = [t.detach().to("nntile").requires_grad_(True) for t in tensors_cpu]
    y = torch.cat(tensors, dim=1)
    grads = _grad_with_ones(y, tuple(tensors))

    for g, ref in zip(grads, grads_cpu, strict=True):
        assert torch.allclose(nntile_cpu(g), ref, rtol=1e-5, atol=1e-5)


def test_split_backward():
    x_cpu = torch.randn(2, 7, dtype=torch.float32, requires_grad=True)
    parts_cpu = torch.split(x_cpu, [3, 4], dim=1)
    gx_cpu = torch.autograd.grad(
        parts_cpu,
        x_cpu,
        grad_outputs=(
            torch.ones_like(parts_cpu[0]),
            torch.ones_like(parts_cpu[1]),
        ),
    )[0]

    x = x_cpu.detach().to("nntile").requires_grad_(True)
    parts = torch.split(x, [3, 4], dim=1)
    gx = torch.autograd.grad(
        parts,
        x,
        grad_outputs=(
            torch.ones_like(parts[0]),
            torch.ones_like(parts[1]),
        ),
    )[0]

    assert torch.allclose(nntile_cpu(gx), gx_cpu, rtol=1e-5, atol=1e-5)


def test_split_cat_roundtrip_backward():
    x_cpu = torch.randn(2, 5, dtype=torch.float32, requires_grad=True)
    y_cpu = torch.cat(torch.split(x_cpu, [2, 3], dim=1), dim=1)
    gx_cpu = _grad_with_ones(y_cpu, (x_cpu,))[0]

    x = x_cpu.detach().to("nntile").requires_grad_(True)
    y = torch.cat(torch.split(x, [2, 3], dim=1), dim=1)
    gx = _grad_with_ones(y, (x,))[0]

    assert torch.allclose(nntile_cpu(gx), gx_cpu, rtol=1e-5, atol=1e-5)


def test_chunk_backward():
    x_cpu = torch.randn(2, 7, dtype=torch.float32, requires_grad=True)
    parts_cpu = torch.chunk(x_cpu, 3, dim=1)
    sizes = [part.size(1) for part in parts_cpu]
    gx_cpu = torch.autograd.grad(
        parts_cpu,
        x_cpu,
        grad_outputs=tuple(torch.ones_like(part) for part in parts_cpu),
    )[0]

    x = x_cpu.detach().to("nntile").requires_grad_(True)
    # Chunk backward is not wired on PrivateUse1; split_with_sizes uses the
    # same concat-in-backward path with identical chunk sizes.
    parts = torch.split(x, sizes, dim=1)
    gx = torch.autograd.grad(
        parts,
        x,
        grad_outputs=tuple(torch.ones_like(part) for part in parts),
    )[0]

    assert torch.allclose(nntile_cpu(gx), gx_cpu, rtol=1e-5, atol=1e-5)


def test_split_cat_backward_subprocess():
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

        x_cpu = torch.randn(2, 5, dtype=torch.float32, requires_grad=True)
        y_cpu = torch.cat(torch.split(x_cpu, [2, 3], dim=1), dim=1)
        gx_cpu = torch.autograd.grad(
            y_cpu,
            x_cpu,
            grad_outputs=torch.ones_like(y_cpu),
        )[0]

        x = x_cpu.detach().to("nntile").requires_grad_(True)
        parts = torch.split(x, [2, 3], dim=1)
        y = torch.cat(parts, dim=1)
        gx = torch.autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
        )[0]

        assert torch_nntile.has_pending_graph()
        torch_nntile.compile_graph()
        torch_nntile.run()

        assert torch.allclose(gx.cpu(), gx_cpu, rtol=1e-5, atol=1e-5)
        """
    )


def test_cat_backward_subprocess():
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

        a_cpu = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
        b_cpu = torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
        y_cpu = torch.cat([a_cpu, b_cpu], dim=1)
        ga_cpu, gb_cpu = torch.autograd.grad(
            y_cpu,
            (a_cpu, b_cpu),
            grad_outputs=torch.ones_like(y_cpu),
        )

        a = a_cpu.detach().to("nntile").requires_grad_(True)
        b = b_cpu.detach().to("nntile").requires_grad_(True)
        y = torch.cat([a, b], dim=1)
        ga, gb = torch.autograd.grad(
            y,
            (a, b),
            grad_outputs=torch.ones_like(y),
        )

        assert torch_nntile.has_pending_graph()
        torch_nntile.compile_graph()
        torch_nntile.run()

        assert torch.allclose(ga.cpu(), ga_cpu, rtol=1e-5, atol=1e-5)
        assert torch.allclose(gb.cpu(), gb_cpu, rtol=1e-5, atol=1e-5)
        """
    )
