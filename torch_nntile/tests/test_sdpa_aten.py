# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_sdpa_aten.py
# F.scaled_dot_product_attention on device=nntile via ATen overrideable.

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from conftest import nntile_cpu
from torch_nntile.nn import sdpa_eager

import torch_nntile


def _reference_sdpa_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    head_size = q.shape[-1]
    scale = 1.0 / math.sqrt(float(head_size))
    scores = q @ k.transpose(-2, -1) * scale
    if is_causal:
        q_seq = q.size(-2)
        k_seq = k.size(-2)
        causal = torch.ones(q_seq, k_seq, dtype=torch.bool)
        causal = torch.tril(causal)
        while causal.dim() < scores.dim():
            causal = causal.unsqueeze(0)
        expand = list(scores.shape[:-2]) + [q_seq, k_seq]
        scores = torch.where(
            causal.expand(expand),
            scores,
            torch.full_like(scores, -math.inf),
        )
    elif attn_mask is not None:
        mask = attn_mask
        if mask.dtype != torch.bool:
            mask = mask > -1e20
        while mask.dim() < scores.dim():
            mask = mask.unsqueeze(0)
        expand = list(scores.shape[:-2]) + [mask.size(-2), mask.size(-1)]
        mask = mask.expand(expand)
        scores = torch.where(
            mask,
            scores,
            torch.full_like(scores, -math.inf),
        )
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 8, 16),
        (4, 2, 8, 16),
        (4, 8, 16),
    ],
)
def test_fsdpa_forward_matches_reference(shape):
    torch.manual_seed(0)
    q_cpu = torch.randn(*shape)
    k_cpu = torch.randn(*shape)
    v_cpu = torch.randn(*shape)
    ref = _reference_sdpa_pytorch(q_cpu, k_cpu, v_cpu)

    q = q_cpu.to("nntile")
    k = k_cpu.to("nntile")
    v = v_cpu.to("nntile")
    out = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(nntile_cpu(out), ref, rtol=1e-4, atol=1e-4)
    assert not torch_nntile.has_pending_graph()


def test_fsdpa_backward_matches_reference():
    shape = (2, 4, 8, 16)
    torch.manual_seed(1)
    q_cpu = torch.randn(*shape, requires_grad=True)
    k_cpu = torch.randn(*shape, requires_grad=True)
    v_cpu = torch.randn(*shape, requires_grad=True)
    ref = _reference_sdpa_pytorch(q_cpu, k_cpu, v_cpu)
    grad_out = torch.randn_like(ref)
    ref.backward(grad_out)

    q = q_cpu.detach().to("nntile").requires_grad_(True)
    k = k_cpu.detach().to("nntile").requires_grad_(True)
    v = v_cpu.detach().to("nntile").requires_grad_(True)
    out = F.scaled_dot_product_attention(q, k, v)
    gq, gk, gv = torch.autograd.grad(
        out,
        (q, k, v),
        grad_outputs=grad_out.to("nntile"),
    )

    assert torch.allclose(nntile_cpu(gq), q_cpu.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(nntile_cpu(gk), k_cpu.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(nntile_cpu(gv), v_cpu.grad, rtol=1e-4, atol=1e-4)


def test_fsdpa_is_causal_matches_reference():
    shape = (2, 4, 8, 16)
    torch.manual_seed(2)
    q_cpu = torch.randn(*shape)
    k_cpu = torch.randn(*shape)
    v_cpu = torch.randn(*shape)
    ref = _reference_sdpa_pytorch(q_cpu, k_cpu, v_cpu, is_causal=True)

    out = F.scaled_dot_product_attention(
        q_cpu.to("nntile"),
        k_cpu.to("nntile"),
        v_cpu.to("nntile"),
        is_causal=True,
    )
    assert torch.allclose(nntile_cpu(out), ref, rtol=1e-4, atol=1e-4)


def test_fsdpa_bool_mask_matches_reference():
    shape = (2, 4, 8, 16)
    seq = shape[-2]
    torch.manual_seed(3)
    q_cpu = torch.randn(*shape)
    k_cpu = torch.randn(*shape)
    v_cpu = torch.randn(*shape)
    mask = torch.zeros(seq, seq, dtype=torch.bool)
    for query in range(seq):
        for key in range(seq):
            mask[query, key] = key <= query

    ref = _reference_sdpa_pytorch(q_cpu, k_cpu, v_cpu, attn_mask=mask)
    out = F.scaled_dot_product_attention(
        q_cpu.to("nntile"),
        k_cpu.to("nntile"),
        v_cpu.to("nntile"),
        attn_mask=mask,
    )
    assert torch.allclose(nntile_cpu(out), ref, rtol=1e-4, atol=1e-4)


def test_fsdpa_float_finfo_mask_matches_reference():
    """HF-style additive mask with finfo.min for masked positions."""
    shape = (2, 4, 8, 16)
    seq = shape[-2]
    torch.manual_seed(5)
    q_cpu = torch.randn(*shape)
    k_cpu = torch.randn(*shape)
    v_cpu = torch.randn(*shape)
    mask = torch.zeros(seq, seq, dtype=torch.float32)
    for query in range(seq):
        for key in range(seq):
            if key <= query:
                mask[query, key] = 0.0
            else:
                mask[query, key] = torch.finfo(torch.float32).min

    ref = _reference_sdpa_pytorch(q_cpu, k_cpu, v_cpu, attn_mask=mask)
    out = F.scaled_dot_product_attention(
        q_cpu.to("nntile"),
        k_cpu.to("nntile"),
        v_cpu.to("nntile"),
        attn_mask=mask,
    )
    assert torch.allclose(nntile_cpu(out), ref, rtol=1e-4, atol=1e-4)


def test_fsdpa_broadcast_4d_mask_matches_reference():
    """Broadcastable ``[1, 1, q_seq, k_seq]`` mask (HF-style)."""
    shape = (2, 4, 8, 16)
    seq = shape[-2]
    torch.manual_seed(6)
    q_cpu = torch.randn(*shape)
    k_cpu = torch.randn(*shape)
    v_cpu = torch.randn(*shape)
    mask_2d = torch.zeros(seq, seq, dtype=torch.bool)
    for query in range(seq):
        for key in range(seq):
            mask_2d[query, key] = key <= query
    mask = mask_2d.view(1, 1, seq, seq)

    ref = _reference_sdpa_pytorch(q_cpu, k_cpu, v_cpu, attn_mask=mask)
    out = F.scaled_dot_product_attention(
        q_cpu.to("nntile"),
        k_cpu.to("nntile"),
        v_cpu.to("nntile"),
        attn_mask=mask,
    )
    assert torch.allclose(nntile_cpu(out), ref, rtol=1e-4, atol=1e-4)


def test_fsdpa_rejects_non_broadcastable_batched_mask():
    shape = (2, 4, 8, 16)
    seq = shape[-2]
    mask = torch.zeros(2, seq, seq, dtype=torch.bool)
    mask[0].fill_(True)
    mask[1].fill_(False)
    q = torch.randn(*shape).to("nntile")
    k = torch.randn(*shape).to("nntile")
    v = torch.randn(*shape).to("nntile")
    with pytest.raises(RuntimeError, match="broadcast"):
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def test_fsdpa_rejects_dropout():
    q = torch.randn(2, 4, 8, 16).to("nntile")
    k = torch.randn(2, 4, 8, 16).to("nntile")
    v = torch.randn(2, 4, 8, 16).to("nntile")
    with pytest.raises(RuntimeError):
        F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)


def test_fsdpa_rejects_custom_scale():
    q = torch.randn(2, 4, 8, 16).to("nntile")
    k = torch.randn(2, 4, 8, 16).to("nntile")
    v = torch.randn(2, 4, 8, 16).to("nntile")
    with pytest.raises(RuntimeError):
        F.scaled_dot_product_attention(q, k, v, scale=0.5)


def test_sdpa_eager_uses_fsdpa_path():
    shape = (2, 8, 16, 4)
    torch.manual_seed(4)
    q_cpu = torch.randn(*shape)
    k_cpu = torch.randn(*shape)
    v_cpu = torch.randn(*shape)

    q_ker = q_cpu.permute(3, 0, 1, 2).contiguous()
    k_ker = k_cpu.permute(3, 0, 1, 2).contiguous()
    v_ker = v_cpu.permute(3, 0, 1, 2).contiguous()
    ref = _reference_sdpa_pytorch(q_ker, k_ker, v_ker)

    out = sdpa_eager(
        q_cpu.to("nntile"),
        k_cpu.to("nntile"),
        v_cpu.to("nntile"),
        batch_ndim=2,
    )
    assert torch.allclose(
        nntile_cpu(out), ref.permute(1, 2, 3, 0), rtol=1e-4, atol=1e-4
    )
