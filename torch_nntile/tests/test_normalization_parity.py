# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_normalization_parity.py
# LayerNorm / RMSNorm parity: CPU PyTorch vs nntile.

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from conftest import nntile_cpu

import torch_nntile


@pytest.mark.parametrize("shape", [(4, 8), (2, 3, 8)])
def test_layer_norm_forward_matches_cpu(shape):
    torch.manual_seed(0)
    feat = shape[-1]
    x_cpu = torch.randn(*shape)
    ln = nn.LayerNorm(feat)
    ln.weight.data.normal_(mean=1.0, std=0.1)
    ln.bias.data.normal_(mean=0.0, std=0.1)

    with torch.no_grad():
        y_cpu = ln(x_cpu)

    ln_nnt = nn.LayerNorm(feat)
    ln_nnt.load_state_dict(ln.state_dict())
    ln_nnt = ln_nnt.to("nntile")
    x_nnt = x_cpu.to("nntile")

    with torch.no_grad():
        y_nnt = nntile_cpu(ln_nnt(x_nnt))

    assert y_nnt.shape == y_cpu.shape
    assert torch.allclose(y_nnt, y_cpu, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(4, 8), (2, 3, 8)])
def test_layer_norm_backward_matches_cpu(shape):
    torch.manual_seed(1)
    feat = shape[-1]
    x_cpu = torch.randn(*shape, requires_grad=True)
    ln = nn.LayerNorm(feat)
    ln.weight.data.normal_(mean=1.0, std=0.1)
    ln.bias.data.normal_(mean=0.0, std=0.1)

    y_cpu = ln(x_cpu)
    y_cpu.backward(torch.ones_like(y_cpu))

    ln_nnt = nn.LayerNorm(feat)
    ln_nnt.load_state_dict(ln.state_dict())
    ln_nnt = ln_nnt.to("nntile")

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    y_nnt = ln_nnt(x_nnt)
    y_nnt.backward(torch.ones_like(y_nnt))

    assert torch.allclose(
        nntile_cpu(x_nnt.grad), x_cpu.grad, rtol=1e-4, atol=1e-4
    )
    assert torch.allclose(
        nntile_cpu(ln_nnt.weight.grad), ln.weight.grad, rtol=1e-4, atol=1e-4
    )
    assert torch.allclose(
        nntile_cpu(ln_nnt.bias.grad), ln.bias.grad, rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("shape", [(4, 8), (2, 3, 8)])
def test_rms_norm_forward_matches_cpu(shape):
    torch.manual_seed(2)
    feat = shape[-1]
    x_cpu = torch.randn(*shape)
    rms = nn.RMSNorm(feat, eps=1e-6)
    rms.weight.data.normal_(mean=1.0, std=0.1)

    with torch.no_grad():
        y_cpu = rms(x_cpu)

    rms_nnt = nn.RMSNorm(feat, eps=1e-6)
    rms_nnt.load_state_dict(rms.state_dict())
    rms_nnt = rms_nnt.to("nntile")
    x_nnt = x_cpu.to("nntile")

    with torch.no_grad():
        y_nnt = nntile_cpu(rms_nnt(x_nnt))

    assert y_nnt.shape == y_cpu.shape
    assert torch.allclose(y_nnt, y_cpu, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(4, 8), (2, 3, 8)])
def test_rms_norm_backward_matches_cpu(shape):
    torch.manual_seed(3)
    feat = shape[-1]
    x_cpu = torch.randn(*shape, requires_grad=True)
    rms = nn.RMSNorm(feat, eps=1e-6)
    rms.weight.data.normal_(mean=1.0, std=0.1)

    y_cpu = rms(x_cpu)
    y_cpu.backward(torch.ones_like(y_cpu))

    rms_nnt = nn.RMSNorm(feat, eps=1e-6)
    rms_nnt.load_state_dict(rms.state_dict())
    rms_nnt = rms_nnt.to("nntile")

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    y_nnt = rms_nnt(x_nnt)
    y_nnt.backward(torch.ones_like(y_nnt))

    assert torch.allclose(
        nntile_cpu(x_nnt.grad), x_cpu.grad, rtol=1e-4, atol=1e-4
    )
    assert torch.allclose(
        nntile_cpu(rms_nnt.weight.grad), rms.weight.grad, rtol=1e-4, atol=1e-4
    )


def test_functional_layer_norm_matches_cpu():
    x_cpu = torch.randn(3, 5)
    weight = torch.ones(5)
    bias = torch.zeros(5)
    y_cpu = F.layer_norm(x_cpu, (5,), weight, bias, 1e-5)

    x_nnt = x_cpu.to("nntile")
    w_nnt = weight.to("nntile")
    b_nnt = bias.to("nntile")
    y_nnt = nntile_cpu(F.layer_norm(x_nnt, (5,), w_nnt, b_nnt, 1e-5))

    assert torch.allclose(y_nnt, y_cpu, rtol=1e-4, atol=1e-4)


def test_functional_rms_norm_matches_cpu():
    x_cpu = torch.randn(3, 5)
    weight = torch.ones(5)
    y_cpu = F.rms_norm(x_cpu, (5,), weight, 1e-6)

    x_nnt = x_cpu.to("nntile")
    w_nnt = weight.to("nntile")
    y_nnt = nntile_cpu(F.rms_norm(x_nnt, (5,), w_nnt, 1e-6))

    assert torch.allclose(y_nnt, y_cpu, rtol=1e-4, atol=1e-4)


def test_rms_norm_without_weight_matches_cpu():
    x_cpu = torch.randn(3, 5)
    y_cpu = F.rms_norm(x_cpu, (5,), None, 1e-6)

    x_nnt = x_cpu.to("nntile")
    y_nnt = nntile_cpu(F.rms_norm(x_nnt, (5,), None, 1e-6))

    assert torch.allclose(y_nnt, y_cpu, rtol=1e-4, atol=1e-4)


def test_rms_norm_without_weight_backward_matches_cpu():
    x_cpu = torch.randn(3, 5, requires_grad=True)
    y_cpu = F.rms_norm(x_cpu, (5,), None, 1e-6)
    y_cpu.backward(torch.ones_like(y_cpu))

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    y_nnt = F.rms_norm(x_nnt, (5,), None, 1e-6)
    y_nnt.backward(torch.ones_like(y_nnt))

    assert torch.allclose(
        nntile_cpu(x_nnt.grad), x_cpu.grad, rtol=1e-4, atol=1e-4
    )
