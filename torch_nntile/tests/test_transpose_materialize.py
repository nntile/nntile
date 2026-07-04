# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_transpose_materialize.py
# HF-style transpose sequences on device=nntile (materialized aten::transpose.int).

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_nntile")

from torch_nntile import _C

_PKG_ROOT = Path(__file__).resolve().parent.parent


def _gpt2_attention_layout(
    x: torch.Tensor,
    n_heads: int,
    head_dim: int,
    scale: float,
    w: torch.Tensor,
) -> torch.Tensor:
    """Eager GPT-2-style Q layout + matmul with transposed K."""
    batch, seq, hidden = x.shape
    q = torch.matmul(x, w)
    states = q.view(batch, seq, n_heads, head_dim).transpose(1, 2)
    k = states
    attn_weights = torch.matmul(states, k.transpose(-1, -2)) * scale
    return attn_weights


def _llama_attention_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Eager Llama-style matmul with key transpose on axes 2 and 3."""
    return torch.matmul(q, k.transpose(2, 3)) / (head_dim**0.5)


def test_transpose_materialize_forward_parity():
    torch.manual_seed(7)
    x_cpu = torch.randn(2, 8, 4, 16)
    x_nnt = x_cpu.to("nntile")
    y_cpu = x_cpu.transpose(1, 2)
    y_nnt = x_nnt.transpose(1, 2).cpu()
    assert y_nnt.is_contiguous()
    torch.testing.assert_close(y_nnt, y_cpu, rtol=1e-5, atol=1e-5)


def test_transpose_last_two_axes_parity():
    torch.manual_seed(8)
    x_cpu = torch.randn(2, 4, 8, 16)
    x_nnt = x_cpu.to("nntile")
    y_cpu = x_cpu.transpose(-1, -2)
    y_nnt = x_nnt.transpose(-1, -2).cpu()
    torch.testing.assert_close(y_nnt, y_cpu, rtol=1e-5, atol=1e-5)


def test_transpose_backward_parity():
    torch.manual_seed(9)
    x_cpu = torch.randn(2, 8, 4, 16, requires_grad=True)
    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    y_cpu = x_cpu.transpose(1, 2)
    grad = torch.randn_like(y_cpu)
    y_cpu.backward(grad)
    y_nnt = x_nnt.transpose(1, 2)
    y_nnt.backward(grad.to("nntile"))
    torch.testing.assert_close(x_nnt.grad.cpu(), x_cpu.grad, rtol=1e-5, atol=1e-5)


def test_gpt2_attention_transpose_sequence_parity():
    torch.manual_seed(10)
    batch, seq, n_heads, head_dim = 2, 8, 4, 16
    hidden = n_heads * head_dim
    x_cpu = torch.randn(batch, seq, hidden)
    w_cpu = torch.randn(hidden, hidden)
    x_nnt = x_cpu.to("nntile")
    w_nnt = w_cpu.to("nntile")
    scale = head_dim**-0.5
    out_cpu = _gpt2_attention_layout(x_cpu, n_heads, head_dim, scale, w_cpu)
    out_nnt = _gpt2_attention_layout(x_nnt, n_heads, head_dim, scale, w_nnt).cpu()
    torch.testing.assert_close(out_nnt, out_cpu, rtol=1e-4, atol=1e-4)


def test_llama_attention_transpose_sequence_parity():
    torch.manual_seed(11)
    bsz, q_len, n_heads, head_dim = 2, 8, 4, 16
    q_cpu = torch.randn(bsz, n_heads, q_len, head_dim)
    k_cpu = torch.randn(bsz, n_heads, q_len, head_dim)
    q_nnt = q_cpu.to("nntile")
    k_nnt = k_cpu.to("nntile")
    out_cpu = _llama_attention_layout(q_cpu, k_cpu, head_dim)
    out_nnt = _llama_attention_layout(q_nnt, k_nnt, head_dim).cpu()
    torch.testing.assert_close(out_nnt, out_cpu, rtol=1e-4, atol=1e-4)


def test_view_transpose_contiguous_sequence_parity():
    torch.manual_seed(12)
    batch, seq, n_heads, head_dim = 2, 8, 4, 16
    hidden = n_heads * head_dim
    x_cpu = torch.randn(batch, seq, hidden)
    x_nnt = x_cpu.to("nntile")
    y_cpu = (
        x_cpu.view(batch, seq, n_heads, head_dim)
        .transpose(1, 2)
        .transpose(1, 2)
        .reshape(batch, seq, hidden)
        .contiguous()
    )
    y_nnt = (
        x_nnt.view(batch, seq, n_heads, head_dim)
        .transpose(1, 2)
        .transpose(1, 2)
        .reshape(batch, seq, hidden)
        .contiguous()
        .cpu()
    )
    torch.testing.assert_close(y_nnt, y_cpu, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)
def test_transpose_graph_mode_deferred():
    repo = Path(__file__).resolve().parents[2]
    build_lib = repo / "build" / "nntile"
    starpu_lib = "/opt/starpu/lib"
    env = dict(**__import__("os").environ)
    ld = env.get("LD_LIBRARY_PATH", "")
    for part in (str(build_lib), starpu_lib):
        if part not in ld.split(":"):
            ld = f"{part}:{ld}" if ld else part
    env["LD_LIBRARY_PATH"] = ld
    env["PYTHONPATH"] = f"{_PKG_ROOT}:{env.get('PYTHONPATH', '')}"
    script = textwrap.dedent(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="graph"
        )
        torch_nntile.restrict_cpu()
        x = torch.randn(2, 8, 4, 16).to("nntile")
        y = x.transpose(1, 2)
        assert torch_nntile.has_pending_graph()
        z = y.transpose(-1, -2)
        assert torch_nntile.has_pending_graph()
        torch_nntile.execute()
        assert not torch_nntile.has_pending_graph()
        ref = x.cpu().transpose(1, 2).transpose(-1, -2)
        torch.testing.assert_close(z.cpu(), ref, rtol=1e-5, atol=1e-5)
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
            f"graph subprocess failed ({proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
