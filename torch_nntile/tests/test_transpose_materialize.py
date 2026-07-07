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
from conftest import nntile_cpu

_PKG_ROOT = Path(__file__).resolve().parent.parent


def test_transpose_materialize_forward_parity():
    torch.manual_seed(7)
    x_cpu = torch.randn(2, 8, 4, 16)
    x_nnt = x_cpu.to("nntile")
    y_cpu = x_cpu.transpose(1, 2)
    y_nnt = nntile_cpu(x_nnt.transpose(1, 2))
    assert y_nnt.is_contiguous()
    torch.testing.assert_close(y_nnt, y_cpu, rtol=1e-5, atol=1e-5)


def test_transpose_last_two_axes_parity():
    torch.manual_seed(8)
    x_cpu = torch.randn(2, 4, 8, 16)
    x_nnt = x_cpu.to("nntile")
    y_cpu = x_cpu.transpose(-1, -2)
    y_nnt = nntile_cpu(x_nnt.transpose(-1, -2))
    torch.testing.assert_close(y_nnt, y_cpu, rtol=1e-5, atol=1e-5)


def test_transpose_backward_parity():
    torch.manual_seed(9)
    x_cpu = torch.randn(2, 8, 4, 16, requires_grad=True)
    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    y_cpu = x_cpu.transpose(1, 2)
    grad = torch.randn_like(y_cpu)
    y_cpu.backward(grad)
    y_nnt = x_nnt.transpose(1, 2)
    y_nnt.backward(grad.contiguous().to("nntile"))
    torch.testing.assert_close(nntile_cpu(x_nnt.grad), x_cpu.grad, rtol=1e-5, atol=1e-5)


def test_gpt2_view_transpose_head_layout_parity():
    """GPT-2: hidden -> view(batch, seq, n_heads, head_dim) -> transpose(1, 2)."""
    torch.manual_seed(10)
    batch, seq, n_heads, head_dim = 2, 8, 4, 16
    hidden = n_heads * head_dim
    q_cpu = torch.randn(batch, seq, hidden)
    q_nnt = q_cpu.to("nntile")
    states_cpu = q_cpu.view(batch, seq, n_heads, head_dim).transpose(1, 2)
    states_nnt = nntile_cpu(q_nnt.view(batch, seq, n_heads, head_dim).transpose(1, 2))
    assert states_nnt.is_contiguous()
    torch.testing.assert_close(states_nnt, states_cpu, rtol=1e-5, atol=1e-5)


def test_gpt2_key_transpose_for_attn_weights_parity():
    """GPT-2: key.transpose(-1, -2) on [B, H, S, D] for matmul layout."""
    torch.manual_seed(11)
    states_cpu = torch.randn(2, 4, 8, 16)
    states_nnt = states_cpu.to("nntile")
    key_t_cpu = states_cpu.transpose(-1, -2)
    key_t_nnt = nntile_cpu(states_nnt.transpose(-1, -2))
    torch.testing.assert_close(key_t_nnt, key_t_cpu, rtol=1e-5, atol=1e-5)


def test_gpt2_attn_output_transpose_reshape_parity():
    """GPT-2: attn_output.transpose(1, 2).reshape(batch, seq, -1).contiguous()."""
    torch.manual_seed(12)
    batch, seq, n_heads, head_dim = 2, 8, 4, 16
    attn_cpu = torch.randn(batch, n_heads, seq, head_dim)
    attn_nnt = attn_cpu.to("nntile")
    out_cpu = attn_cpu.transpose(1, 2).reshape(batch, seq, -1).contiguous()
    out_nnt = nntile_cpu(attn_nnt.transpose(1, 2).reshape(batch, seq, -1).contiguous())
    torch.testing.assert_close(out_nnt, out_cpu, rtol=1e-5, atol=1e-5)


def test_llama_query_transpose_parity():
    """Llama: q_proj(h).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)."""
    torch.manual_seed(13)
    bsz, q_len, n_heads, head_dim = 2, 8, 4, 16
    hidden = n_heads * head_dim
    q_cpu = torch.randn(bsz, q_len, hidden)
    q_nnt = q_cpu.to("nntile")
    query_cpu = q_cpu.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    query_nnt = nntile_cpu(q_nnt.view(bsz, q_len, n_heads, head_dim).transpose(1, 2))
    torch.testing.assert_close(query_nnt, query_cpu, rtol=1e-5, atol=1e-5)


def test_llama_key_transpose_for_attn_weights_parity():
    """Llama: key_states.transpose(2, 3) on [B, H, S, D]."""
    torch.manual_seed(14)
    k_cpu = torch.randn(2, 4, 8, 16)
    k_nnt = k_cpu.to("nntile")
    k_t_cpu = k_cpu.transpose(2, 3)
    k_t_nnt = nntile_cpu(k_nnt.transpose(2, 3))
    torch.testing.assert_close(k_t_nnt, k_t_cpu, rtol=1e-5, atol=1e-5)


def test_view_transpose_contiguous_sequence_parity():
    torch.manual_seed(15)
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
    y_nnt = nntile_cpu(
        x_nnt.view(batch, seq, n_heads, head_dim)
        .transpose(1, 2)
        .transpose(1, 2)
        .reshape(batch, seq, hidden)
        .contiguous()
    )
    torch.testing.assert_close(y_nnt, y_cpu, rtol=1e-5, atol=1e-5)


def test_transpose_then_contiguous_is_noop_when_materialized():
    torch.manual_seed(16)
    x_cpu = torch.randn(2, 8, 4, 16)
    x_nnt = x_cpu.to("nntile")
    y_cpu = x_cpu.transpose(1, 2).contiguous()
    y_nnt = nntile_cpu(x_nnt.transpose(1, 2).contiguous())
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
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
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
