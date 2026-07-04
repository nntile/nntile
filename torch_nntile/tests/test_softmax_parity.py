# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_softmax_parity.py
# Softmax forward/backward parity: CPU PyTorch vs nntile.

import torch
import pytest

import torch_nntile
from torch_nntile import _C


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


@pytest.fixture
def random_input():
    torch.manual_seed(0)
    return torch.randn(4, 8)


def test_softmax_forward_matches_cpu(random_input):
    x_cpu = random_input
    y_cpu = torch.nn.functional.softmax(x_cpu, dim=-1)

    x_nnt = x_cpu.to("nntile")
    y_nnt = torch.nn.functional.softmax(x_nnt, dim=-1).cpu()

    assert torch.allclose(y_nnt, y_cpu, rtol=1e-4, atol=1e-4)


def test_softmax_forward_dim0(random_input):
    x_cpu = random_input
    y_cpu = torch.nn.functional.softmax(x_cpu, dim=0)

    x_nnt = x_cpu.to("nntile")
    y_nnt = torch.nn.functional.softmax(x_nnt, dim=0).cpu()

    assert torch.allclose(y_nnt, y_cpu, rtol=1e-4, atol=1e-4)


def test_softmax_backward_matches_cpu(random_input):
    x_cpu = random_input.clone().requires_grad_(True)
    y_cpu = torch.nn.functional.softmax(x_cpu, dim=-1)
    y_cpu.backward(torch.ones_like(y_cpu))

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    y_nnt = torch.nn.functional.softmax(x_nnt, dim=-1)
    y_nnt.backward(torch.ones(y_nnt.shape, device="cpu").to("nntile"))

    assert torch.allclose(x_nnt.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


def test_softmax_backward_dim0(random_input):
    x_cpu = random_input.clone().requires_grad_(True)
    y_cpu = torch.nn.functional.softmax(x_cpu, dim=0)
    y_cpu.backward(torch.ones_like(y_cpu))

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    y_nnt = torch.nn.functional.softmax(x_nnt, dim=0)
    y_nnt.backward(torch.ones(y_nnt.shape, device="cpu").to("nntile"))

    assert torch.allclose(x_nnt.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


def test_nn_softmax_module(random_input):
    x_cpu = random_input.clone().requires_grad_(True)
    module_cpu = torch.nn.Softmax(dim=1)
    y_cpu = module_cpu(x_cpu)
    y_cpu.backward(torch.ones_like(y_cpu))

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    module_nnt = torch.nn.Softmax(dim=1)
    y_nnt = module_nnt(x_nnt)
    y_nnt.backward(torch.ones(y_nnt.shape, device="cpu").to("nntile"))

    assert torch.allclose(y_nnt.cpu(), y_cpu, rtol=1e-4, atol=1e-4)
    assert torch.allclose(x_nnt.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)
