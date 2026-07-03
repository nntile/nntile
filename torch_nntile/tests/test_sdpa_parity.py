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
            runtime_mode="eager",
        )
    torch_nntile.restrict_cpu()
    yield


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
        (4, 2, 8, 16),
        (2, 3, 6, 8),
        (1, 1, 4, 8),
    ],
)
def test_sdpa_forward_matches_reference(shape):
    torch.manual_seed(0)
    q_cpu = torch.randn(*shape)
    k_cpu = torch.randn(*shape)
    v_cpu = torch.randn(*shape)
    ref = _reference_sdpa_eager(q_cpu, k_cpu, v_cpu, None)

    q = q_cpu.to("nntile")
    k = k_cpu.to("nntile")
    v = v_cpu.to("nntile")
    out = sdpa_eager(q, k, v, batch_ndim=2)
    assert torch.allclose(out.cpu(), ref, rtol=1e-4, atol=1e-4)
    assert not torch_nntile.has_pending_graph()


@pytest.mark.parametrize("shape", [(4, 2, 8, 16), (2, 3, 6, 8)])
def test_sdpa_backward_matches_reference(shape):
    torch.manual_seed(1)
    q_cpu = torch.randn(*shape, requires_grad=True)
    k_cpu = torch.randn(*shape, requires_grad=True)
    v_cpu = torch.randn(*shape, requires_grad=True)
    ref = _reference_sdpa_eager(q_cpu, k_cpu, v_cpu, None)
    grad_out = torch.randn_like(ref)
    ref.backward(grad_out)

    q = q_cpu.detach().to("nntile").requires_grad_(True)
    k = k_cpu.detach().to("nntile").requires_grad_(True)
    v = v_cpu.detach().to("nntile").requires_grad_(True)
    out = sdpa_eager(q, k, v, batch_ndim=2)
    out.backward(grad_out.to("nntile"))

    assert torch.allclose(q.grad.cpu(), q_cpu.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(k.grad.cpu(), k_cpu.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(v.grad.cpu(), v_cpu.grad, rtol=1e-4, atol=1e-4)


def test_sdpa_forward_with_mask_matches_reference():
    shape = (2, 2, 6, 8)
    seq = shape[2]
    torch.manual_seed(2)
    q_cpu = torch.randn(*shape)
    k_cpu = torch.randn(*shape)
    v_cpu = torch.randn(*shape)
    mask = torch.zeros(seq, seq, dtype=torch.bool)
    for key in range(seq):
        for query in range(seq):
            mask[key, query] = key <= query

    ref = _reference_sdpa_eager(q_cpu, k_cpu, v_cpu, mask)
    out = sdpa_eager(
        q_cpu.to("nntile"),
        k_cpu.to("nntile"),
        v_cpu.to("nntile"),
        mask,
        batch_ndim=2,
    )
    assert torch.allclose(out.cpu(), ref, rtol=1e-4, atol=1e-4)


def test_sdpa_module_forward():
    mod = SDPA(batch_ndim=2)
    q = torch.randn(2, 1, 4, 8).to("nntile")
    k = torch.randn(2, 1, 4, 8).to("nntile")
    v = torch.randn(2, 1, 4, 8).to("nntile")
    out = mod(q, k, v)
    assert out.shape == q.shape


def test_sdpa_rejects_cpu_tensors():
    q = torch.randn(2, 1, 4, 8)
    k = torch.randn(2, 1, 4, 8)
    v = torch.randn(2, 1, 4, 8)
    with pytest.raises(ValueError, match="nntile"):
        sdpa_eager(q, k, v)


def test_sdpa_rejects_graph_mode():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = f"""
import sys
sys.path.insert(0, {str(root)!r})
import torch
import torch_nntile
from torch_nntile.nn import sdpa_eager

torch_nntile.init_context(
    ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="graph"
)
torch_nntile.restrict_cpu()
q = torch.randn(2, 1, 4, 8).to("nntile")
k = torch.randn(2, 1, 4, 8).to("nntile")
v = torch.randn(2, 1, 4, 8).to("nntile")
try:
    sdpa_eager(q, k, v)
except RuntimeError as exc:
    assert "runtime_mode='eager'" in str(exc)
else:
    raise AssertionError("expected RuntimeError")
"""
    subprocess.check_call([sys.executable, "-c", script])
