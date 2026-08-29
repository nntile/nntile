# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_aten_ops_parity.py
# Standard ATen ops on device=nntile vs CPU, forward + backward.

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from conftest import nntile_cpu
from parity_helpers import assert_aten_op_forward_backward, assert_close

import torch_nntile

_RTOL = 1e-4
_ATOL = 1e-4
_BWD_RTOL = 1e-3
_BWD_ATOL = 1e-3


def _seeded(*shapes: tuple[int, ...]) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [
        torch.randn(shape, dtype=torch.float32, requires_grad=True)
        for shape in shapes
    ]


# Ops with working PrivateUse1 forward + autograd on nntile.
# Keep this matrix aligned with registered aten PrivateUse1 compute kernels.
_FWD_BWD_CASES = [
    (
        "add",
        lambda: _seeded((4, 6), (4, 6)),
        lambda a, b: a + b,
    ),
    (
        "add_alpha",
        lambda: _seeded((3, 5), (3, 5)),
        lambda a, b: torch.add(a, b, alpha=1.5),
    ),
    (
        "mul",
        lambda: _seeded((4, 6), (4, 6)),
        lambda a, b: a * b,
    ),
    (
        "mul_scaled",
        lambda: _seeded((4, 6)),
        lambda a: a * torch.full_like(a, 1.5),
    ),
    (
        "relu",
        lambda: _seeded((4, 8)),
        torch.relu,
    ),
    (
        "silu",
        lambda: _seeded((4, 8)),
        F.silu,
    ),
    (
        "gelu_none",
        lambda: _seeded((4, 8)),
        lambda x: F.gelu(x, approximate="none"),
    ),
    (
        "gelu_tanh",
        lambda: _seeded((4, 8)),
        lambda x: F.gelu(x, approximate="tanh"),
    ),
    (
        "softmax_last",
        lambda: _seeded((3, 8)),
        lambda x: F.softmax(x, dim=-1),
    ),
    (
        "mm",
        lambda: _seeded((5, 7), (7, 4)),
        torch.mm,
    ),
    (
        "bmm",
        lambda: _seeded((2, 5, 7), (2, 7, 4)),
        torch.bmm,
    ),
    (
        "addmm",
        lambda: _seeded((4,), (5, 7), (7, 4)),
        lambda bias, mat1, mat2: torch.addmm(bias, mat1, mat2),
    ),
    (
        "linear",
        lambda: _seeded((3, 8), (5, 8), (5,)),
        lambda x, w, b: F.linear(x, w, b),
    ),
    (
        "linear_no_bias",
        lambda: _seeded((3, 8), (5, 8)),
        lambda x, w: F.linear(x, w),
    ),
    (
        "cat_dim0",
        lambda: _seeded((2, 4), (3, 4)),
        lambda a, b: torch.cat([a, b], dim=0),
    ),
    (
        "cat_dim1",
        lambda: _seeded((2, 3), (2, 4)),
        lambda a, b: torch.cat([a, b], dim=1),
    ),
    (
        "split_cat_roundtrip",
        lambda: _seeded((2, 7)),
        lambda x: torch.cat(torch.split(x, [3, 4], dim=1), dim=1),
    ),
    (
        "transpose",
        lambda: _seeded((3, 5)),
        lambda x: x.transpose(0, 1).contiguous(),
    ),
    (
        "t",
        lambda: _seeded((3, 5)),
        lambda x: x.t().contiguous(),
    ),
    (
        "view_reshape",
        lambda: _seeded((2, 3, 4)),
        lambda x: x.reshape(6, 4),
    ),
    (
        "exp",
        lambda: _seeded((4, 8)),
        torch.exp,
    ),
    (
        "cos",
        lambda: _seeded((4, 8)),
        torch.cos,
    ),
    (
        "sin",
        lambda: _seeded((4, 8)),
        torch.sin,
    ),
]


@pytest.mark.parametrize(
    "name,make_inputs,op",
    _FWD_BWD_CASES,
    ids=[c[0] for c in _FWD_BWD_CASES],
)
def test_aten_op_matches_cpu_forward_backward(name, make_inputs, op):
    del name
    inputs = make_inputs()
    assert_aten_op_forward_backward(
        op,
        inputs_cpu=inputs,
        rtol=_RTOL,
        atol=_ATOL,
        bwd_rtol=_BWD_RTOL,
        bwd_atol=_BWD_ATOL,
    )


def test_embedding_matches_cpu_forward_backward():
    torch.manual_seed(0)
    weight = torch.randn(16, 8, dtype=torch.float32, requires_grad=True)
    indices = torch.randint(0, 16, (2, 5), dtype=torch.long)

    # F.embedding(input, weight) - indices first.
    assert_aten_op_forward_backward(
        F.embedding,
        inputs_cpu=[indices, weight],
        check_input_grads=[False, True],
        rtol=_RTOL,
        atol=_ATOL,
        bwd_rtol=_BWD_RTOL,
        bwd_atol=_BWD_ATOL,
    )


def test_layer_norm_matches_cpu_forward_backward():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(8, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(8, dtype=torch.float32, requires_grad=True)

    def op(inp, w, b):
        return F.layer_norm(inp, (8,), weight=w, bias=b, eps=1e-5)

    assert_aten_op_forward_backward(
        op,
        inputs_cpu=[x, weight, bias],
        rtol=_RTOL,
        atol=_ATOL,
        bwd_rtol=_BWD_RTOL,
        bwd_atol=_BWD_ATOL,
    )


def test_layer_norm_no_affine_matches_cpu_forward_backward():
    """DiT AdaLN uses LayerNorm(elementwise_affine=False)."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, dtype=torch.float32, requires_grad=True)

    def op(inp):
        return F.layer_norm(inp, (8,), weight=None, bias=None, eps=1e-6)

    assert_aten_op_forward_backward(
        op,
        inputs_cpu=[x],
        rtol=_RTOL,
        atol=_ATOL,
        bwd_rtol=_BWD_RTOL,
        bwd_atol=_BWD_ATOL,
    )


@pytest.mark.parametrize(
    "name,make_inputs,op",
    [
        (
            "hypot_forward",
            lambda: _seeded((4, 6), (4, 6)),
            torch.hypot,
        ),
        (
            "sum_last_forward",
            lambda: _seeded((3, 5, 7)),
            lambda x: torch.sum(x, dim=-1),
        ),
        (
            "chunk_forward",
            lambda: _seeded((2, 9)),
            lambda x: torch.cat(torch.chunk(x, 3, dim=1), dim=1),
        ),
        (
            "repeat_forward",
            lambda: _seeded((2, 3)),
            lambda x: x.repeat(2, 1),
        ),
        (
            "linalg_vector_norm_forward",
            lambda: _seeded((3, 5)),
            lambda x: torch.linalg.vector_norm(x, ord=2, dim=-1),
        ),
        (
            "narrow_forward",
            lambda: _seeded((2, 8)),
            lambda x: x.narrow(1, 2, 4),
        ),
    ],
    ids=[
        "hypot",
        "sum_last",
        "chunk",
        "repeat",
        "vector_norm",
        "narrow",
    ],
)
def test_aten_op_matches_cpu_forward_only(name, make_inputs, op):
    """Forward parity for ops whose nntile backward is not ready yet."""
    del name
    inputs = [t.detach() for t in make_inputs()]
    y_ref = op(*inputs)
    y_nnt = op(*(t.to("nntile") for t in inputs))
    assert_close(y_nnt, y_ref, rtol=_RTOL, atol=_ATOL)
    # Touch nntile_cpu path for graph flush consistency.
    assert nntile_cpu(y_nnt).shape == y_ref.shape


def test_arange_as_first_graph_op_matches_cpu():
    """Factory ``aten::arange`` must compile even as the first graph op."""
    torch_nntile.reset_graph_session()
    ref = torch.arange(8)
    got = torch.arange(8, device="nntile")
    torch.testing.assert_close(nntile_cpu(got), ref)
    got_start = torch.arange(0, 8, device="nntile")
    torch.testing.assert_close(nntile_cpu(got_start), torch.arange(0, 8))
    ref_f = torch.arange(4, dtype=torch.float32)
    got_f = torch.arange(4, dtype=torch.float32, device="nntile")
    torch.testing.assert_close(nntile_cpu(got_f), ref_f)


def test_strided_view_ops_match_cpu_forward_backward():
    """Non-contiguous views must match CPU/CUDA (no wrapper densify)."""
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
    y = x[:, ::2]

    for op in (torch.relu, F.silu, lambda t: F.gelu(t, approximate="tanh")):
        assert_aten_op_forward_backward(
            op,
            inputs_cpu=[y],
            rtol=_RTOL,
            atol=_ATOL,
            bwd_rtol=_BWD_RTOL,
            bwd_atol=_BWD_ATOL,
        )

    a = torch.randn(4, 6, dtype=torch.float32, requires_grad=True)
    b = torch.randn(4, 6, dtype=torch.float32, requires_grad=True)
    assert_aten_op_forward_backward(
        lambda u, v: u * v,
        inputs_cpu=[a[:, ::2], b[:, ::2]],
        rtol=_RTOL,
        atol=_ATOL,
        bwd_rtol=_BWD_RTOL,
        bwd_atol=_BWD_ATOL,
    )


def test_sdpa_matches_cpu_forward_backward():
    """ATen SDPA overrideable path (fwd + bwd) vs CPU."""
    torch.manual_seed(0)
    b, h, s, d = 1, 2, 4, 8
    q, k, v = _seeded((b, h, s, d), (b, h, s, d), (b, h, s, d))

    def op(qq, kk, vv):
        return F.scaled_dot_product_attention(qq, kk, vv, is_causal=False)

    assert_aten_op_forward_backward(
        op,
        inputs_cpu=[q, k, v],
        rtol=_RTOL,
        atol=_ATOL,
        bwd_rtol=_BWD_RTOL,
        bwd_atol=_BWD_ATOL,
    )


@pytest.mark.skipif(
    not getattr(torch_nntile, "TORCH_NATIVE_OPS", False),
    reason="torch-native aten ops not built",
)
def test_aten_relu_records_torch_native_graph():
    from classic_graph import assert_torch_native_graph

    torch_nntile.reset_graph_session()
    x_n = (
        torch.randn(4, 8, dtype=torch.float32)
        .to("nntile")
        .requires_grad_(True)
    )
    y_n = torch.relu(x_n)
    ones = torch.ones(tuple(y_n.shape), dtype=y_n.dtype)
    torch.autograd.grad(y_n, x_n, ones.to(y_n.device))
    assert_torch_native_graph()


@pytest.mark.skipif(
    not getattr(torch_nntile, "TORCH_NATIVE_OPS", False),
    reason="torch-native aten ops not built",
)
def test_stock_linear_records_torch_native_graph():
    from classic_graph import assert_torch_native_graph

    torch_nntile.reset_graph_session()
    layer = torch.nn.Linear(8, 5, bias=True).float().to("nntile")
    x = torch.randn(3, 8, dtype=torch.float32).to("nntile").requires_grad_(
        True
    )
    y = layer(x)
    ones = torch.ones(tuple(y.shape), dtype=y.dtype)
    torch.autograd.grad(y, x, ones.to(y.device))
    assert_torch_native_graph()

