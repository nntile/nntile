# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_transpose_materialize.py
# HF-style transpose / narrow / split as zero-copy views on device=nntile.

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch

pytest.importorskip("torch_nntile")

from conftest import nntile_cpu, subprocess_environ


def test_transpose_view_forward_parity():
    torch.manual_seed(7)
    x_cpu = torch.randn(2, 8, 4, 16)
    x_nnt = x_cpu.to("nntile")
    y_cpu = x_cpu.transpose(1, 2)
    y_nnt_view = x_nnt.transpose(1, 2)
    assert not y_nnt_view.is_contiguous()
    y_nnt = nntile_cpu(y_nnt_view)
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
    (gx_cpu,) = torch.autograd.grad(y_cpu, x_cpu, grad_outputs=grad)
    y_nnt = x_nnt.transpose(1, 2)
    (gx_nnt,) = torch.autograd.grad(
        y_nnt,
        x_nnt,
        grad_outputs=grad.contiguous().to("nntile"),
    )
    torch.testing.assert_close(
        nntile_cpu(gx_nnt), gx_cpu, rtol=1e-5, atol=1e-5
    )


def test_gpt2_view_transpose_head_layout_parity():
    """GPT-2: hidden -> view(batch, seq, n_heads, head_dim) -> transpose(1, 2)."""
    torch.manual_seed(10)
    batch, seq, n_heads, head_dim = 2, 8, 4, 16
    hidden = n_heads * head_dim
    q_cpu = torch.randn(batch, seq, hidden)
    q_nnt = q_cpu.to("nntile")
    states_cpu = q_cpu.view(batch, seq, n_heads, head_dim).transpose(1, 2)
    states_nnt_view = q_nnt.view(batch, seq, n_heads, head_dim).transpose(
        1, 2
    )
    assert not states_nnt_view.is_contiguous()
    states_nnt = nntile_cpu(states_nnt_view)
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
    """GPT-2: attn_output.transpose(1, 2).contiguous().reshape(...)."""
    torch.manual_seed(12)
    batch, seq, n_heads, head_dim = 2, 8, 4, 16
    attn_cpu = torch.randn(batch, n_heads, seq, head_dim)
    attn_nnt = attn_cpu.to("nntile")
    out_cpu = attn_cpu.transpose(1, 2).contiguous().reshape(batch, seq, -1)
    out_nnt = nntile_cpu(
        attn_nnt.transpose(1, 2).contiguous().reshape(batch, seq, -1)
    )
    torch.testing.assert_close(out_nnt, out_cpu, rtol=1e-5, atol=1e-5)


def test_llama_query_transpose_parity():
    """Llama: q_proj(h).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)."""
    torch.manual_seed(13)
    bsz, q_len, n_heads, head_dim = 2, 8, 4, 16
    hidden = n_heads * head_dim
    q_cpu = torch.randn(bsz, q_len, hidden)
    q_nnt = q_cpu.to("nntile")
    query_cpu = q_cpu.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    query_nnt = nntile_cpu(
        q_nnt.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    )
    torch.testing.assert_close(query_nnt, query_cpu, rtol=1e-5, atol=1e-5)


def test_llama_key_transpose_for_attn_weights_parity():
    """Llama: key_states.transpose(2, 3) on [B, H, S, D]."""
    torch.manual_seed(14)
    k_cpu = torch.randn(2, 4, 8, 16)
    k_nnt = k_cpu.to("nntile")
    k_t_cpu = k_cpu.transpose(2, 3)
    k_t_nnt = nntile_cpu(k_nnt.transpose(2, 3))
    torch.testing.assert_close(k_t_nnt, k_t_cpu, rtol=1e-5, atol=1e-5)


def test_view_transpose_reshape_sequence_parity():
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
    )
    torch.testing.assert_close(y_nnt, y_cpu, rtol=1e-5, atol=1e-5)


def test_contiguous_densifies_transpose_view():
    torch.manual_seed(16)
    x_cpu = torch.randn(2, 8, 4, 16)
    x_nnt = x_cpu.to("nntile")
    y_cpu = x_cpu.transpose(1, 2).contiguous()
    y_nnt_view = x_nnt.transpose(1, 2)
    assert not y_nnt_view.is_contiguous()
    y_nnt = nntile_cpu(y_nnt_view.contiguous())
    assert y_nnt.is_contiguous()
    torch.testing.assert_close(y_nnt, y_cpu, rtol=1e-5, atol=1e-5)


def test_permute_is_zero_copy_view():
    torch.manual_seed(17)
    x_cpu = torch.randn(2, 8, 4, 16)
    x_nnt = x_cpu.to("nntile")
    y_cpu = x_cpu.permute(0, 2, 1, 3)
    y_nnt_view = x_nnt.permute(0, 2, 1, 3)
    assert not y_nnt_view.is_contiguous()
    torch.testing.assert_close(
        nntile_cpu(y_nnt_view), y_cpu, rtol=1e-5, atol=1e-5
    )


def test_split_narrow_offset_view_parity():
    """split/narrow must keep storage_offset (not densify)."""
    torch.manual_seed(18)
    x_cpu = torch.randn(2, 8, 48)
    x_nnt = x_cpu.to("nntile")
    parts_cpu = torch.split(x_cpu, 16, dim=2)
    parts_nnt = torch.split(x_nnt, 16, dim=2)
    assert len(parts_nnt) == 3
    for pc, pn in zip(parts_cpu, parts_nnt):
        assert pn.storage_offset() == pc.storage_offset()
        assert pn.stride() == pc.stride()
        torch.testing.assert_close(
            nntile_cpu(pn), pc, rtol=1e-5, atol=1e-5
        )
        # Second half of each split chunk via narrow.
        n_cpu = pc.narrow(2, 4, 8)
        n_nnt = pn.narrow(2, 4, 8)
        assert n_nnt.storage_offset() == n_cpu.storage_offset()
        torch.testing.assert_close(
            nntile_cpu(n_nnt), n_cpu, rtol=1e-5, atol=1e-5
        )


def test_transpose_view_needs_no_graph_op():
    env = subprocess_environ()
    script = textwrap.dedent(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()
        x = torch.randn(2, 8, 4, 16).to("nntile")
        # Ingress scatter may still be pending; flush before view checks.
        torch_nntile.compile_graph()
        torch_nntile.run()
        y = x.transpose(1, 2)
        assert not y.is_contiguous()
        assert not torch_nntile.has_pending_graph()
        z = y.transpose(-1, -2)
        assert not torch_nntile.has_pending_graph()
        ref = x.cpu().transpose(1, 2).transpose(-1, -2)
        torch.testing.assert_close(z.cpu(), ref, rtol=1e-5, atol=1e-5)
        # Densify records a Copy into the graph.
        c = z.contiguous()
        assert torch_nntile.has_pending_graph()
        torch_nntile.compile_graph()
        torch_nntile.run()
        torch.testing.assert_close(
            c.cpu(), ref.contiguous(), rtol=1e-5, atol=1e-5
        )
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
