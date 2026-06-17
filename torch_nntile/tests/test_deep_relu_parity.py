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
    yield


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
    model_nnt.load_state_dict(model_cpu.state_dict())
    model_nnt = model_nnt.to("nntile")
    x_nnt = x_cpu.to("nntile")

    with torch.no_grad():
        y_nnt = model_nnt(x_nnt).cpu()

    assert y_nnt.shape == y_cpu.shape
    assert torch.allclose(y_nnt, y_cpu, rtol=1e-4, atol=1e-4)


def test_linear_relu_layer_matches_cpu():
    x_cpu = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    weight = torch.tensor([[0.25, -0.5, 1.0], [2.0, 0.0, -1.0]])

    y_cpu = torch.nn.functional.relu(x_cpu @ weight.t())

    x_nnt = x_cpu.to("nntile")
    w_nnt = weight.to("nntile")
    y_nnt = torch.nn.functional.relu(
        torch.nn.functional.linear(x_nnt, w_nnt, None)
    ).cpu()

    assert torch.allclose(y_nnt, y_cpu, rtol=1e-4, atol=1e-4)


def test_linear_relu_layer_backward_matches_cpu():
    x_cpu = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]], requires_grad=True)
    weight = torch.tensor(
        [[0.25, -0.5, 1.0], [2.0, 0.0, -1.0]],
        requires_grad=True,
    )

    y_cpu = torch.nn.functional.relu(x_cpu @ weight.t())
    y_cpu.backward(torch.ones_like(y_cpu))

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    w_nnt = weight.detach().to("nntile").requires_grad_(True)
    y_nnt = torch.nn.functional.relu(
        torch.nn.functional.linear(x_nnt, w_nnt, None)
    )
    y_nnt.backward(torch.ones(y_nnt.shape, device="cpu").to("nntile"))

    assert torch.allclose(x_nnt.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(w_nnt.grad.cpu(), weight.grad, rtol=1e-4, atol=1e-4)


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
    model_nnt.load_state_dict(model_cpu.state_dict())
    model_nnt = model_nnt.to("nntile")

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    y_nnt = model_nnt(x_nnt)
    grad_out = torch.ones(y_nnt.shape, device="cpu").to("nntile")

    params_nnt = list(model_nnt.parameters())
    gx_nnt, *grads_nnt = torch.autograd.grad(
        y_nnt, [x_nnt, *params_nnt], grad_outputs=grad_out
    )
    gx_cpu, *grads_cpu = torch.autograd.grad(
        y_cpu, [x_cpu, *model_cpu.parameters()], grad_outputs=grad_out_cpu
    )

    for g_nnt, g_cpu in zip(grads_nnt, grads_cpu):
        assert torch.allclose(g_nnt.cpu(), g_cpu, rtol=1e-4, atol=1e-4)
    assert torch.allclose(gx_nnt.cpu(), gx_cpu, rtol=1e-4, atol=1e-4)
