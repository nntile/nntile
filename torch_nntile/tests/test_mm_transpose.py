# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_mm_transpose.py

import pytest
import torch

pytest.importorskip("torch_nntile")


def test_mm_transpose_view_parity():
    torch.manual_seed(2)
    a_cpu = torch.randn(3, 4)
    b_cpu = torch.randn(3, 5)
    a_nnt = a_cpu.to("nntile")
    b_nnt = b_cpu.to("nntile")
    out_cpu = torch.mm(a_cpu.t(), b_cpu)
    out_nnt = torch.mm(a_nnt.t(), b_nnt).cpu()
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
    torch.testing.assert_close(a_nnt.grad.cpu(), a_cpu.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(b_nnt.grad.cpu(), b_cpu.grad, rtol=1e-5, atol=1e-5)


def test_contiguous_permute_matmul():
    torch.manual_seed(4)
    x_cpu = torch.randn(2, 3, 4)
    w_cpu = torch.randn(3, 5)
    x_nnt = x_cpu.to("nntile")
    w_nnt = w_cpu.to("nntile")
    out_cpu = torch.matmul(x_cpu.permute(0, 2, 1).contiguous(), w_cpu)
    x_perm_nnt = x_nnt.detach().permute(0, 2, 1).contiguous()
    out_nnt = torch.matmul(x_perm_nnt, w_nnt).cpu()
    torch.testing.assert_close(out_nnt, out_cpu, rtol=1e-5, atol=1e-5)
