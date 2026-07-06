# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_sdpa_parity.py
# SDPA eager parity vs NNTile-layout reference implementation.

from __future__ import annotations

import math

import pytest
import torch

import torch_nntile
from torch_nntile import _C
from torch_nntile.nn import SDPA, sdpa_eager
from conftest import nntile_cpu


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


@pytest.fixture(scope="module", autouse=True)
def _nntile_context_no_fallback():
    if not _C.has_libnntile():
        return
    if torch_nntile.is_cpu_fallback_enabled():
        pytest.skip(
            "context has CPU fallback enabled; rebuild with cpu_fallback=False"
        )
    if not torch_nntile.is_context_initialized():
        torch_nntile.init_context(
            ncpu=1,
            ncuda=0,
            verbose=0,
            cpu_fallback=False,
        )
    torch_nntile.restrict_cpu()
    yield


def _projection_to_kernel_layout(x: torch.Tensor) -> torch.Tensor:
    """``[batch, seq, head_size, n_heads]`` -> ``[n_heads, batch, seq, head_size]``."""
    return x.permute(3, 0, 1, 2).contiguous()


def _kernel_to_projection_layout(x: torch.Tensor) -> torch.Tensor:
    """``[n_heads, batch, seq, head_size]`` -> ``[batch, seq, head_size, n_heads]``."""
    return x.permute(1, 2, 3, 0).contiguous()


def _reference_sdpa_projection(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    out = _reference_sdpa_eager(
        _projection_to_kernel_layout(q),
        _projection_to_kernel_layout(k),
        _projection_to_kernel_layout(v),
        mask,
    )
    return _kernel_to_projection_layout(out)


def _reference_sdpa_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    head_size = q.shape[-1]
    scale = 1.0 / math.sqrt(float(head_size))
    scores = torch.einsum("...ed,...cd->...ce", k, q) * scale
    if mask is not None:
        mask_expanded = mask.to(dtype=torch.bool, device=scores.device)
        while mask_expanded.dim() < scores.dim():
            mask_expanded = mask_expanded.unsqueeze(0)
        expand_shape = list(scores.shape[:-2]) + [
            mask_expanded.size(-2),
            mask_expanded.size(-1),
        ]
        mask_expanded = mask_expanded.expand(expand_shape)
        scores = torch.where(
            mask_expanded,
            scores,
            torch.full_like(scores, -math.inf),
        )
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("...ce,...ed->...cd", attn, v)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 8, 16, 4),
        (3, 6, 8, 2),
        (1, 4, 8, 1),
    ],
)
def test_sdpa_forward_matches_reference(shape):
    torch.manual_seed(0)
    q_cpu = torch.randn(*shape)
    k_cpu = torch.randn(*shape)
    v_cpu = torch.randn(*shape)
    ref = _reference_sdpa_projection(q_cpu, k_cpu, v_cpu, None)

    q = q_cpu.to("nntile")
    k = k_cpu.to("nntile")
    v = v_cpu.to("nntile")
    out = sdpa_eager(q, k, v, batch_ndim=2)
    assert torch.allclose(nntile_cpu(out), ref, rtol=1e-4, atol=1e-4)
    assert not torch_nntile.has_pending_graph()


@pytest.mark.parametrize("shape", [(2, 8, 16, 4), (3, 6, 8, 2)])
def test_sdpa_backward_matches_reference(shape):
    torch.manual_seed(1)
    q_cpu = torch.randn(*shape, requires_grad=True)
    k_cpu = torch.randn(*shape, requires_grad=True)
    v_cpu = torch.randn(*shape, requires_grad=True)
    ref = _reference_sdpa_projection(q_cpu, k_cpu, v_cpu, None)
    grad_out = torch.randn_like(ref)
    ref.backward(grad_out)

    q = q_cpu.detach().to("nntile").requires_grad_(True)
    k = k_cpu.detach().to("nntile").requires_grad_(True)
    v = v_cpu.detach().to("nntile").requires_grad_(True)
    out = sdpa_eager(q, k, v, batch_ndim=2)
    out.backward(grad_out.to("nntile"))

    assert torch.allclose(nntile_cpu(q.grad), q_cpu.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(nntile_cpu(k.grad), k_cpu.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(nntile_cpu(v.grad), v_cpu.grad, rtol=1e-4, atol=1e-4)


def test_sdpa_forward_with_mask_matches_reference():
    shape = (2, 6, 8, 2)
    seq = shape[1]
    torch.manual_seed(2)
    q_cpu = torch.randn(*shape)
    k_cpu = torch.randn(*shape)
    v_cpu = torch.randn(*shape)
    mask = torch.zeros(seq, seq, dtype=torch.bool)
    for query in range(seq):
        for key in range(seq):
            mask[query, key] = key <= query

    ref = _reference_sdpa_projection(q_cpu, k_cpu, v_cpu, mask)
    out = sdpa_eager(
        q_cpu.to("nntile"),
        k_cpu.to("nntile"),
        v_cpu.to("nntile"),
        mask.to("nntile"),
        batch_ndim=2,
    )
    assert torch.allclose(nntile_cpu(out), ref, rtol=1e-4, atol=1e-4)


def test_sdpa_mask_axis_order_matches_causal_layout():
    """Mask dim0=query, dim1=key (libnntile sdpa_causal_mask_bool_fill layout)."""
    shape = (1, 5, 8, 1)
    seq = shape[1]
    torch.manual_seed(7)
    q_cpu = torch.randn(*shape)
    k_cpu = torch.randn(*shape)
    v_cpu = torch.randn(*shape)
    mask = torch.zeros(seq, seq, dtype=torch.bool)
    for query in range(seq):
        for key in range(seq):
            mask[query, key] = key <= query

    ref = _reference_sdpa_projection(q_cpu, k_cpu, v_cpu, mask)
    out = sdpa_eager(
        q_cpu.to("nntile"),
        k_cpu.to("nntile"),
        v_cpu.to("nntile"),
        mask.to("nntile"),
        batch_ndim=2,
    )
    assert torch.allclose(nntile_cpu(out), ref, rtol=1e-4, atol=1e-4)

    # Asymmetric pattern: only (query=2, key=3) allowed. Transposed mask must differ.
    mask_sparse = torch.zeros(seq, seq, dtype=torch.bool)
    mask_sparse[2, 3] = True
    out_sparse = sdpa_eager(
        q_cpu.to("nntile"),
        k_cpu.to("nntile"),
        v_cpu.to("nntile"),
        mask_sparse.to("nntile"),
        batch_ndim=2,
    )
    out_transposed_mask = sdpa_eager(
        q_cpu.to("nntile"),
        k_cpu.to("nntile"),
        v_cpu.to("nntile"),
        mask_sparse.t().contiguous().to("nntile"),
        batch_ndim=2,
    )
    assert not torch.allclose(
        nntile_cpu(out_sparse), nntile_cpu(out_transposed_mask), rtol=1e-4, atol=1e-4
    )


def test_sdpa_backward_rejects_mismatched_grad_out():
    shape = (2, 1, 4, 8)
    q = torch.randn(*shape).to("nntile")
    k = torch.randn(*shape).to("nntile")
    v = torch.randn(*shape).to("nntile")
    bad_grad = torch.randn(2, 1, 3, 8).to("nntile")
    with pytest.raises(RuntimeError, match="grad_out shape must match"):
        _C.sdpa_backward(q, k, v, bad_grad, None, 2)


def test_sdpa_module_forward():
    mod = SDPA(batch_ndim=2)
    # Post-GEMM layout: [batch, seq, head_size, n_heads]
    q = torch.randn(1, 4, 8, 2).to("nntile")
    k = torch.randn(1, 4, 8, 2).to("nntile")
    v = torch.randn(1, 4, 8, 2).to("nntile")
    out = mod(q, k, v)
    assert out.shape == q.shape


def test_sdpa_rejects_cpu_tensors():
    q = torch.randn(1, 4, 8, 2)
    k = torch.randn(1, 4, 8, 2)
    v = torch.randn(1, 4, 8, 2)
    with pytest.raises(ValueError, match="nntile"):
        sdpa_eager(q, k, v)


def test_sdpa_graph_mode_deferred_until_execute():
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    repo = Path(__file__).resolve().parents[2]
    env = dict(**__import__("os").environ)
    build_lib = repo / "build" / "nntile"
    starpu_lib = "/opt/starpu/lib"
    ld = env.get("LD_LIBRARY_PATH", "")
    for part in (str(build_lib), starpu_lib):
        if part not in ld.split(":"):
            ld = f"{part}:{ld}" if ld else part
    env["LD_LIBRARY_PATH"] = ld
    env["PYTHONPATH"] = f"{root}:{env.get('PYTHONPATH', '')}"

    script = textwrap.dedent(
        """
        import torch
        import torch_nntile
        from torch_nntile.nn import sdpa_eager

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()

        shape = (1, 4, 8, 2)
        q_cpu = torch.randn(*shape, requires_grad=True)
        k_cpu = torch.randn(*shape, requires_grad=True)
        v_cpu = torch.randn(*shape, requires_grad=True)
        q_ker = q_cpu.permute(3, 0, 1, 2).contiguous()
        k_ker = k_cpu.permute(3, 0, 1, 2).contiguous()
        v_ker = v_cpu.permute(3, 0, 1, 2).contiguous()
        ref = torch.einsum(
            "...ce,...ed->...cd",
            torch.softmax(
                torch.einsum("...ed,...cd->...ce", k_ker, q_ker)
                * (shape[-2] ** -0.5),
                dim=-1,
            ),
            v_ker,
        ).permute(1, 2, 3, 0).contiguous()
        grad_out = torch.randn_like(ref)
        ref.backward(grad_out)

        q = q_cpu.detach().to("nntile").requires_grad_(True)
        k = k_cpu.detach().to("nntile").requires_grad_(True)
        v = v_cpu.detach().to("nntile").requires_grad_(True)
        out = sdpa_eager(q, k, v, batch_ndim=2)
        assert torch_nntile.has_pending_graph()
        out.backward(grad_out.to("nntile"))
        assert torch_nntile.has_pending_graph()
        torch_nntile.execute()
        assert not torch_nntile.has_pending_graph()
        assert torch.allclose(out.detach().cpu(), ref, rtol=1e-4, atol=1e-4)
        assert torch.allclose(q.grad.cpu(), q_cpu.grad, rtol=1e-4, atol=1e-4)
        assert torch.allclose(k.grad.cpu(), k_cpu.grad, rtol=1e-4, atol=1e-4)
        assert torch.allclose(v.grad.cpu(), v_cpu.grad, rtol=1e-4, atol=1e-4)
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
