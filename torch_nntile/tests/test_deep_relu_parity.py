# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_deep_relu_parity.py
# DeepReLU forward parity: CPU PyTorch vs nntile (no CPU fallback).

import torch
import pytest

import torch_nntile
from torch_nntile import _C
from torch_nntile.models import DeepReLU
from conftest import nntile_cpu


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


def test_deep_relu_forward_matches_cpu():
    torch.manual_seed(0)
    model_cpu = DeepReLU.tiny()
    model_cpu.init_kaiming_uniform_(seed=42)

    batch = 32
    generator = torch.Generator()
    generator.manual_seed(123)
    x_cpu = torch.randn(batch, model_cpu.input_dim, generator=generator)

    with torch.no_grad():
        y_cpu = model_cpu(x_cpu)

    model_nnt = DeepReLU.tiny()
    with torch.no_grad():
        model_nnt.load_state_dict(model_cpu.state_dict())
        model_nnt = model_nnt.to("nntile")
        x_nnt = x_cpu.to("nntile")

    with torch.no_grad():
        y_nnt = nntile_cpu(model_nnt(x_nnt))

    assert y_nnt.shape == y_cpu.shape
    assert torch.allclose(y_nnt, y_cpu, rtol=1e-4, atol=1e-4)


def test_linear_relu_layer_matches_cpu():
    x_cpu = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    weight = torch.tensor([[0.25, -0.5, 1.0], [2.0, 0.0, -1.0]])

    y_cpu = torch.nn.functional.relu(x_cpu @ weight.t())

    with torch.no_grad():
        x_nnt = x_cpu.to("nntile")
        w_nnt = weight.to("nntile")
    y_nnt = nntile_cpu(
        torch.nn.functional.relu(
            torch.nn.functional.linear(x_nnt, w_nnt, None)
        )
    )

    assert torch.allclose(y_nnt, y_cpu, rtol=1e-4, atol=1e-4)


def test_linear_relu_layer_backward_matches_cpu():
    x_cpu = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]], requires_grad=True)
    weight = torch.tensor(
        [[0.25, -0.5, 1.0], [2.0, 0.0, -1.0]],
        requires_grad=True,
    )

    y_cpu = torch.nn.functional.relu(x_cpu @ weight.t())
    grad_out = torch.ones_like(y_cpu)
    y_cpu.backward(grad_out)

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    w_nnt = weight.detach().to("nntile").requires_grad_(True)
    y_nnt = torch.nn.functional.relu(
        torch.nn.functional.linear(x_nnt, w_nnt, None)
    )
    with torch.no_grad():
        grad_out_nnt = grad_out.to("nntile")
    gx_nnt, gw_nnt = torch.autograd.grad(
        y_nnt,
        (x_nnt, w_nnt),
        grad_outputs=grad_out_nnt,
    )

    assert torch.allclose(nntile_cpu(gx_nnt), x_cpu.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(nntile_cpu(gw_nnt), weight.grad, rtol=1e-4, atol=1e-4)


def test_deep_relu_backward_matches_cpu():
    torch.manual_seed(0)
    model_cpu = DeepReLU.tiny()
    model_cpu.init_kaiming_uniform_(seed=42)

    batch = 32
    generator = torch.Generator()
    generator.manual_seed(123)
    x_cpu = torch.randn(
        batch, model_cpu.input_dim, generator=generator, requires_grad=True
    )

    y_cpu = model_cpu(x_cpu)
    grad_out_cpu = torch.ones_like(y_cpu)

    model_nnt = DeepReLU.tiny()
    with torch.no_grad():
        model_nnt.load_state_dict(model_cpu.state_dict())
        model_nnt = model_nnt.to("nntile")

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    y_nnt = model_nnt(x_nnt)
    with torch.no_grad():
        grad_out = torch.ones(y_nnt.shape, device="cpu").to("nntile")

    params_nnt = list(model_nnt.parameters())
    gx_nnt, *grads_nnt = torch.autograd.grad(
        y_nnt, [x_nnt, *params_nnt], grad_outputs=grad_out
    )
    gx_cpu, *grads_cpu = torch.autograd.grad(
        y_cpu, [x_cpu, *model_cpu.parameters()], grad_outputs=grad_out_cpu
    )

    for g_nnt, g_cpu in zip(grads_nnt, grads_cpu):
        assert torch.allclose(nntile_cpu(g_nnt), g_cpu, rtol=1e-4, atol=1e-4)
    assert torch.allclose(nntile_cpu(gx_nnt), gx_cpu, rtol=1e-4, atol=1e-4)
