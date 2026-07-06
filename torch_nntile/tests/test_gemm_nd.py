# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_gemm_nd.py
# N-D GEMM parity tests (GPT-2 attention projection shapes).

from __future__ import annotations

import pytest
import torch

from torch_nntile import _C
from torch_nntile.gemm import gemm

pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


@pytest.fixture(scope="module", autouse=True)
def _init_nntile():
    import torch_nntile

    if not torch_nntile.is_context_initialized():
        torch_nntile.init_context(
            ncpu=1,
            ncuda=0,
            verbose=0,
            cpu_fallback=False,
        )
    torch_nntile.restrict_cpu()
    yield


def _cpu_ref_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    ndim: int,
    batch_ndim: int,
) -> torch.Tensor:
    """Reference via explicit einsum-style reshape (batch_ndim=0 only)."""
    assert batch_ndim == 0
    a_rank = a.dim()
    b_rank = b.dim()
    m_shape = a.shape[: a_rank - ndim]
    k_a = a.shape[a_rank - ndim :]
    k_b = b.shape[:ndim]
    n_shape = b.shape[ndim:]
    assert k_a == k_b
    a_flat = a.reshape(*m_shape, *k_a)
    b_flat = b.reshape(*k_b, *n_shape)
    k = int(torch.tensor(k_a).prod().item())
    m = int(torch.tensor(m_shape).prod().item()) if m_shape else 1
    n = int(torch.tensor(n_shape).prod().item()) if n_shape else 1
    out = a_flat.reshape(m, k) @ b_flat.reshape(k, n)
    return out.reshape(*m_shape, *n_shape)


def test_gemm_qkv_projection_shape():
    bsz, seq, hidden, hs, n_heads = 2, 8, 64, 16, 4
    x = torch.randn(bsz, seq, hidden)
    w = torch.randn(hidden, hs, n_heads)
    x_n = x.to("nntile")
    w_n = w.to("nntile")
    out = gemm(x_n, w_n, ndim=1, batch_ndim=0)
    assert out.shape == (bsz, seq, hs, n_heads)
    ref = _cpu_ref_gemm(x, w, ndim=1, batch_ndim=0)
    assert torch.allclose(out.cpu(), ref, atol=1e-5, rtol=1e-5)


def test_gemm_output_projection_shape():
    bsz, seq, hs, n_heads, hidden = 2, 8, 16, 4, 64
    attn = torch.randn(bsz, seq, hs, n_heads)
    w_o = torch.randn(hs, n_heads, hidden)
    attn_n = attn.to("nntile")
    w_o_n = w_o.to("nntile")
    out = gemm(attn_n, w_o_n, ndim=2, batch_ndim=0)
    assert out.shape == (bsz, seq, hidden)
    ref = _cpu_ref_gemm(attn, w_o, ndim=2, batch_ndim=0)
    assert torch.allclose(out.cpu(), ref, atol=1e-5, rtol=1e-5)


def test_matmul_inferred_qkv():
    bsz, seq, hidden, hs, n_heads = 2, 8, 64, 16, 4
    x = torch.randn(bsz, seq, hidden)
    w = torch.randn(hidden, hs, n_heads)
    out = torch.matmul(x.to("nntile"), w.to("nntile"))
    assert out.shape == (bsz, seq, hs, n_heads)
    ref = _cpu_ref_gemm(x, w, ndim=1, batch_ndim=0)
    assert torch.allclose(out.cpu(), ref, atol=1e-5, rtol=1e-5)


def test_gemm_qkv_backward():
    bsz, seq, hidden, hs, n_heads = 2, 8, 64, 16, 4
    x = torch.randn(bsz, seq, hidden, requires_grad=True)
    w = torch.randn(hidden, hs, n_heads, requires_grad=True)
    x_n = x.detach().to("nntile").requires_grad_(True)
    w_n = w.detach().to("nntile").requires_grad_(True)
    out = gemm(x_n, w_n, ndim=1, batch_ndim=0)
    grad_out = torch.randn_like(out.cpu()).to("nntile")
    out.backward(grad_out)
    ref_out = _cpu_ref_gemm(x, w, ndim=1, batch_ndim=0)
    ref_out.backward(grad_out.cpu())
    assert torch.allclose(x_n.grad.cpu(), x.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(w_n.grad.cpu(), w.grad, atol=1e-4, rtol=1e-4)


def test_gemm_graph_mode_no_view():
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    pkg_root = Path(__file__).resolve().parent.parent
    repo = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import torch
        import torch_nntile
        from torch_nntile.gemm import gemm

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()
        bsz, seq, hidden, hs, n_heads = 2, 4, 32, 8, 4
        x = torch.randn(bsz, seq, hidden).to("nntile")
        w = torch.randn(hidden, hs, n_heads).to("nntile")
        bias = (
            torch.randn(n_heads, hs)
            .transpose(0, 1)
            .view(1, 1, hs, n_heads)
            .expand(bsz, seq, hs, n_heads)
            .contiguous()
            .to("nntile")
        )
        proj = gemm(x, w, ndim=1, batch_ndim=0)
        out = proj + bias
        assert out.shape == (bsz, seq, hs, n_heads)
        torch_nntile.execute()
        """
    )
    import os

    env = dict(os.environ)
    build_lib = repo / "build" / "nntile"
    starpu_lib = "/opt/starpu/lib"
    ld = env.get("LD_LIBRARY_PATH", "")
    for part in (str(build_lib), starpu_lib):
        if part not in ld.split(":"):
            ld = f"{part}:{ld}" if ld else part
    env["LD_LIBRARY_PATH"] = ld
    env["PYTHONPATH"] = f"{pkg_root}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
