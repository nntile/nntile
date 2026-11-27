# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_mm_transpose.py

import pytest
import torch

pytest.importorskip("torch_nntile")
from conftest import nntile_cpu


def test_mm_transpose_view_parity():
    torch.manual_seed(2)
    a_cpu = torch.randn(3, 4)
    b_cpu = torch.randn(3, 5)
    a_nnt = a_cpu.to("nntile")
    b_nnt = b_cpu.to("nntile")
    out_cpu = torch.mm(a_cpu.t(), b_cpu)
    out_nnt = nntile_cpu(torch.mm(a_nnt.t(), b_nnt))
    torch.testing.assert_close(out_nnt, out_cpu, rtol=1e-5, atol=1e-5)


def test_mm_backward_parity():
    torch.manual_seed(3)
    a_cpu = torch.randn(4, 3, requires_grad=True)
    b_cpu = torch.randn(3, 5, requires_grad=True)
    a_nnt = a_cpu.detach().to("nntile").requires_grad_(True)
    b_nnt = b_cpu.detach().to("nntile").requires_grad_(True)
    out_cpu = torch.mm(a_cpu, b_cpu)
    grad_out = torch.ones_like(out_cpu)
    out_cpu.backward(grad_out)
    out_nnt = torch.mm(a_nnt, b_nnt)
    out_nnt.backward(grad_out.to("nntile"))
    torch.testing.assert_close(
        nntile_cpu(a_nnt.grad), a_cpu.grad, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        nntile_cpu(b_nnt.grad), b_cpu.grad, rtol=1e-5, atol=1e-5
    )


def test_permute_zero_copy_view():
    torch.manual_seed(4)
    x_cpu = torch.randn(2, 3, 4)
    x_nnt = x_cpu.to("nntile")
    y_nnt = x_nnt.permute(0, 2, 1)
    assert not y_nnt.is_contiguous()
    torch.testing.assert_close(
        nntile_cpu(y_nnt), x_cpu.permute(0, 2, 1), rtol=1e-5, atol=1e-5
    )


def test_linear_transpose_weight_backward_parity():
    torch.manual_seed(5)
    base = torch.randn(5, 4)
    x_cpu = torch.randn(3, 5, requires_grad=True)
    w_cpu = base.t().contiguous().requires_grad_(True)
    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    # Densify after t(): linear gemm layout expects a dense [out, in]
    # leaf (same as nn.Linear weight storage).
    w_nnt = base.to("nntile").t().contiguous().requires_grad_(True)
    out_cpu = torch.nn.functional.linear(x_cpu, w_cpu)
    grad_out = torch.ones_like(out_cpu)
    out_cpu.backward(grad_out)
    out_nnt = torch.nn.functional.linear(x_nnt, w_nnt)
    out_nnt.backward(grad_out.to("nntile"))
    torch.testing.assert_close(
        nntile_cpu(x_nnt.grad), x_cpu.grad, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        nntile_cpu(w_nnt.grad).contiguous(),
        w_cpu.grad,
        rtol=1e-5,
        atol=1e-5,
    )
