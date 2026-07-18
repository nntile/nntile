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
from conftest import nntile_cpu, subprocess_environ


def _grad_with_ones(
    output: torch.Tensor, inputs: tuple[torch.Tensor, ...]
) -> tuple:
    """Backward via grad_outputs; avoids ``sum`` (not on nntile)."""
    grad_out = torch.ones_like(output)
    return torch.autograd.grad(output, inputs, grad_outputs=grad_out)


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
        torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
        for _ in range(4)
    ]
    y_cpu = torch.cat(tensors_cpu, dim=1)
    grads_cpu = _grad_with_ones(y_cpu, tuple(tensors_cpu))

    tensors = [
        t.detach().to("nntile").requires_grad_(True) for t in tensors_cpu
    ]
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


def test_view_backward_cat_no_resize_warning():
    """View Backward reshape views must densify before SplitBackward cat.

    Reproduces HF GPT-2 attn: heads [B,S,H,D] → View Backward to
    [B,S,H*D] then SplitBackward cat. Without densify, StarPU cat packs
    node shapes and aten::cat_out resize-warns fused → [B,S,3H,D].
    """
    env = subprocess_environ()
    script = textwrap.dedent(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1,
            ncuda=0,
            cpu_fallback=False,
        )
        torch_nntile.restrict_cpu()

        batch, seq, n_heads, head_dim = 1, 8, 4, 16
        hidden = n_heads * head_dim
        qkv_cpu = torch.randn(
            batch, seq, 3 * hidden, dtype=torch.float32, requires_grad=True
        )
        q_c, k_c, v_c = qkv_cpu.split(hidden, dim=2)
        qh_c = q_c.view(batch, seq, n_heads, head_dim)
        kh_c = k_c.view(batch, seq, n_heads, head_dim)
        vh_c = v_c.view(batch, seq, n_heads, head_dim)
        gx_cpu = torch.autograd.grad(
            (qh_c, kh_c, vh_c),
            qkv_cpu,
            grad_outputs=(
                torch.ones_like(qh_c),
                torch.ones_like(kh_c),
                torch.ones_like(vh_c),
            ),
        )[0]

        qkv = qkv_cpu.detach().to("nntile").requires_grad_(True)
        q, k, v = qkv.split(hidden, dim=2)
        qh = q.view(batch, seq, n_heads, head_dim)
        kh = k.view(batch, seq, n_heads, head_dim)
        vh = v.view(batch, seq, n_heads, head_dim)
        gx = torch.autograd.grad(
            (qh, kh, vh),
            qkv,
            grad_outputs=(
                torch.ones_like(qh),
                torch.ones_like(kh),
                torch.ones_like(vh),
            ),
        )[0]
        assert torch_nntile.has_pending_graph()
        torch_nntile.compile_graph()
        torch_nntile.run()
        assert torch.allclose(gx.cpu(), gx_cpu, rtol=1e-5, atol=1e-5)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
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
    if (
        "Resize.cpp" in proc.stderr
        or "resized since it had shape" in proc.stderr
    ):
        raise AssertionError(
            "unexpected Resize.cpp / resize_output warning\n"
            f"stderr:\n{proc.stderr}"
        )
