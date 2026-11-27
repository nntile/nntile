# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_silu_gelu_parity.py
# SiLU and GELU forward/backward parity: CPU PyTorch vs nntile.

import pytest
import torch
from conftest import nntile_cpu

import torch_nntile


@pytest.fixture
def random_input():
    torch.manual_seed(0)
    return torch.randn(4, 8)


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_forward_matches_cpu(random_input, approximate):
    x_cpu = random_input
    y_cpu = torch.nn.functional.gelu(x_cpu, approximate=approximate)

    x_nnt = x_cpu.to("nntile")
    y_nnt = nntile_cpu(
        torch.nn.functional.gelu(x_nnt, approximate=approximate)
    )

    assert torch.allclose(y_nnt, y_cpu, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_backward_matches_cpu(random_input, approximate):
    x_cpu = random_input.clone().requires_grad_(True)
    y_cpu = torch.nn.functional.gelu(x_cpu, approximate=approximate)
    y_cpu.backward(torch.ones_like(y_cpu))

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    y_nnt = torch.nn.functional.gelu(x_nnt, approximate=approximate)
    y_nnt.backward(torch.ones(y_nnt.shape, device="cpu").to("nntile"))

    assert torch.allclose(
        nntile_cpu(x_nnt.grad), x_cpu.grad, rtol=1e-4, atol=1e-4
    )


def test_silu_forward_matches_cpu(random_input):
    x_cpu = random_input
    y_cpu = torch.nn.functional.silu(x_cpu)

    x_nnt = x_cpu.to("nntile")
    y_nnt = nntile_cpu(torch.nn.functional.silu(x_nnt))

    assert torch.allclose(y_nnt, y_cpu, rtol=1e-4, atol=1e-4)


def test_silu_backward_matches_cpu(random_input):
    x_cpu = random_input.clone().requires_grad_(True)
    y_cpu = torch.nn.functional.silu(x_cpu)
    y_cpu.backward(torch.ones_like(y_cpu))

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    y_nnt = torch.nn.functional.silu(x_nnt)
    y_nnt.backward(torch.ones(y_nnt.shape, device="cpu").to("nntile"))

    assert torch.allclose(
        nntile_cpu(x_nnt.grad), x_cpu.grad, rtol=1e-4, atol=1e-4
    )


def test_silu_inplace_matches_cpu(random_input):
    x_cpu = random_input.clone()
    y_cpu = torch.nn.functional.silu(x_cpu)

    x_nnt = random_input.clone().to("nntile")
    torch.nn.functional.silu(x_nnt, inplace=True)

    assert torch.allclose(nntile_cpu(x_nnt), y_cpu, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_inplace_matches_cpu(random_input, approximate):
    x_cpu = random_input.clone()
    y_cpu = torch.nn.functional.gelu(x_cpu, approximate=approximate)

    x_nnt = random_input.clone().to("nntile")
    torch.ops.aten.gelu_(x_nnt, approximate=approximate)

    assert torch.allclose(nntile_cpu(x_nnt), y_cpu, rtol=1e-4, atol=1e-4)


def test_linear_silu_layer_matches_cpu():
    x_cpu = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    weight = torch.tensor([[0.25, -0.5, 1.0], [2.0, 0.0, -1.0]])

    y_cpu = torch.nn.functional.silu(x_cpu @ weight.t())

    x_nnt = x_cpu.to("nntile")
    w_nnt = weight.to("nntile")
    y_nnt = nntile_cpu(
        torch.nn.functional.silu(
            torch.nn.functional.linear(x_nnt, w_nnt, None)
        )
    )

    assert torch.allclose(y_nnt, y_cpu, rtol=1e-4, atol=1e-4)


def test_linear_silu_layer_backward_matches_cpu():
    x_cpu = torch.tensor(
        [[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]], requires_grad=True
    )
    weight = torch.tensor(
        [[0.25, -0.5, 1.0], [2.0, 0.0, -1.0]],
        requires_grad=True,
    )

    y_cpu = torch.nn.functional.silu(x_cpu @ weight.t())
    grad_out = torch.ones_like(y_cpu)
    y_cpu.backward(grad_out)

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    w_nnt = weight.detach().to("nntile").requires_grad_(True)
    y_nnt = torch.nn.functional.silu(
        torch.nn.functional.linear(x_nnt, w_nnt, None)
    )
    gx_nnt, gw_nnt = torch.autograd.grad(
        y_nnt,
        (x_nnt, w_nnt),
        grad_outputs=grad_out.to("nntile"),
    )

    assert torch.allclose(nntile_cpu(gx_nnt), x_cpu.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(
        nntile_cpu(gw_nnt), weight.grad, rtol=1e-4, atol=1e-4
    )


def test_linear_gelu_layer_backward_matches_cpu():
    x_cpu = torch.tensor(
        [[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]], requires_grad=True
    )
    weight = torch.tensor(
        [[0.25, -0.5, 1.0], [2.0, 0.0, -1.0]],
        requires_grad=True,
    )

    y_cpu = torch.nn.functional.gelu(x_cpu @ weight.t())
    grad_out = torch.ones_like(y_cpu)
    y_cpu.backward(grad_out)

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    w_nnt = weight.detach().to("nntile").requires_grad_(True)
    y_nnt = torch.nn.functional.gelu(
        torch.nn.functional.linear(x_nnt, w_nnt, None)
    )
    gx_nnt, gw_nnt = torch.autograd.grad(
        y_nnt,
        (x_nnt, w_nnt),
        grad_outputs=grad_out.to("nntile"),
    )

    assert torch.allclose(nntile_cpu(gx_nnt), x_cpu.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(
        nntile_cpu(gw_nnt), weight.grad, rtol=1e-4, atol=1e-4
    )
