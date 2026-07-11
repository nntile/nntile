# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_linear_bias_parity.py
# Linear + bias forward/backward parity: CPU PyTorch vs nntile.

import pytest
import torch
import torch.nn.functional as F

import torch_nntile
from torch_nntile import _C
from conftest import nntile_cpu

pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


@pytest.mark.parametrize(
    "shape",
    [
        (4, 5),
        (2, 3, 5),
    ],
)
def test_linear_bias_forward_matches_cpu(shape):
    torch.manual_seed(0)
    in_features = shape[-1]
    out_features = 7
    x_cpu = torch.randn(*shape)
    w_cpu = torch.randn(out_features, in_features)
    b_cpu = torch.randn(out_features)

    y_cpu = F.linear(x_cpu, w_cpu, b_cpu)

    with torch.no_grad():
        y_nnt = nntile_cpu(
            F.linear(x_cpu.to("nntile"), w_cpu.to("nntile"), b_cpu.to("nntile"))
        )

    assert y_nnt.shape == y_cpu.shape
    torch.testing.assert_close(y_nnt, y_cpu, rtol=1e-5, atol=1e-5)


def test_linear_bias_backward_matches_cpu():
    """2D full backward (input/weight/bias); ND weight host-copy is pre-existing."""
    torch.manual_seed(1)
    x_cpu = torch.randn(4, 5, requires_grad=True)
    w_cpu = torch.randn(7, 5, requires_grad=True)
    b_cpu = torch.randn(7, requires_grad=True)

    y_cpu = F.linear(x_cpu, w_cpu, b_cpu)
    grad_out = torch.randn_like(y_cpu)
    y_cpu.backward(grad_out)

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    w_nnt = w_cpu.detach().to("nntile").requires_grad_(True)
    b_nnt = b_cpu.detach().to("nntile").requires_grad_(True)
    y_nnt = F.linear(x_nnt, w_nnt, b_nnt)
    with torch.no_grad():
        grad_out_nnt = grad_out.to("nntile")
    gx, gw, gb = torch.autograd.grad(
        y_nnt,
        (x_nnt, w_nnt, b_nnt),
        grad_outputs=grad_out_nnt,
    )

    torch.testing.assert_close(
        nntile_cpu(gx), x_cpu.grad, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        nntile_cpu(gw), w_cpu.grad, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        nntile_cpu(gb), b_cpu.grad, rtol=1e-4, atol=1e-4
    )


def test_linear_bias_nd_grad_bias_matches_cpu():
    """ND activations: forward bias + grad_bias via sum_fiber."""
    torch.manual_seed(2)
    x_cpu = torch.randn(2, 3, 5)
    w_cpu = torch.randn(7, 5)
    b_cpu = torch.randn(7, requires_grad=True)
    grad_out = torch.randn(2, 3, 7)

    y_cpu = F.linear(x_cpu, w_cpu, b_cpu)
    y_cpu.backward(grad_out)

    b_nnt = b_cpu.detach().to("nntile").requires_grad_(True)
    y_nnt = F.linear(x_cpu.to("nntile"), w_cpu.to("nntile"), b_nnt)
    (gb,) = torch.autograd.grad(
        y_nnt,
        (b_nnt,),
        grad_outputs=grad_out.to("nntile"),
    )
    torch.testing.assert_close(
        nntile_cpu(gb), b_cpu.grad, rtol=1e-4, atol=1e-4
    )


def test_linear_none_bias_still_works():
    torch.manual_seed(2)
    x_cpu = torch.randn(3, 5)
    w_cpu = torch.randn(4, 5)
    y_cpu = F.linear(x_cpu, w_cpu, None)
    with torch.no_grad():
        y_nnt = nntile_cpu(
            F.linear(x_cpu.to("nntile"), w_cpu.to("nntile"), None)
        )
    torch.testing.assert_close(y_nnt, y_cpu, rtol=1e-5, atol=1e-5)


def test_linear_bias_shape_mismatch_raises():
    x = torch.randn(2, 5).to("nntile")
    w = torch.randn(4, 5).to("nntile")
    b = torch.randn(3).to("nntile")
    with pytest.raises(RuntimeError, match="bias size"):
        F.linear(x, w, b)


def test_nn_linear_with_bias_matches_cpu():
    torch.manual_seed(3)
    layer = torch.nn.Linear(6, 4, bias=True)
    x_cpu = torch.randn(8, 6)
    with torch.no_grad():
        y_cpu = layer(x_cpu)

    layer_nnt = torch.nn.Linear(6, 4, bias=True)
    layer_nnt.load_state_dict(layer.state_dict())
    layer_nnt = layer_nnt.to("nntile")
    with torch.no_grad():
        y_nnt = nntile_cpu(layer_nnt(x_cpu.to("nntile")))

    torch.testing.assert_close(y_nnt, y_cpu, rtol=1e-5, atol=1e-5)
