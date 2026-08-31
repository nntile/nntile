# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_nn_classic.py
# Classic torch_nntile.nn ops vs CPU formulas.

from __future__ import annotations

import torch
import torch.nn.functional as F
from classic_graph import assert_classic_graph
from conftest import nntile_cpu
from torch_nntile.nn import Embedding, LayerNorm, Linear, ReLU
from torch_nntile.nn.functional import add, gelu, relu
from torch_nntile.nn.sdpa import nntile_model_transpose


def test_nn_linear_relu_matches_cpu():
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    layer = torch.nn.Linear(8, 5, bias=True)
    y_cpu = torch.nn.functional.relu(layer(x))
    with torch.no_grad():
        layer_n = Linear(8, 5, bias=True)
        layer_n.load_state_dict(layer.state_dict())
        layer_n = layer_n.to("nntile")
        act_n = ReLU()
        x_n = x.to("nntile")
    y_n = nntile_cpu(act_n(layer_n(x_n)))
    assert torch.allclose(y_n, y_cpu, rtol=1e-4, atol=1e-4)


def test_nn_layernorm_matches_cpu():
    torch.manual_seed(1)
    x = torch.randn(3, 6, 8)
    ln = LayerNorm(8, eps=1e-5)
    y_cpu = ln(x)
    with torch.no_grad():
        ln_n = LayerNorm(8, eps=1e-5)
        ln_n.load_state_dict(ln.state_dict())
        ln_n = ln_n.to("nntile")
        x_n = x.to("nntile")
    y_n = nntile_cpu(ln_n(x_n))
    assert torch.allclose(y_n, y_cpu, rtol=1e-4, atol=1e-4)


def test_nn_embedding_matches_cpu():
    torch.manual_seed(2)
    weight = torch.randn(16, 8)
    idx = torch.randint(0, 16, (2, 5), dtype=torch.long)
    y_cpu = F.embedding(idx, weight)
    with torch.no_grad():
        emb = Embedding(16, 8)
        emb.weight.data.copy_(weight)
        emb = emb.to("nntile")
        idx_n = idx.to("nntile")
    y_n = nntile_cpu(emb(idx_n))
    assert torch.allclose(y_n, y_cpu, rtol=1e-4, atol=1e-4)


def test_activations_and_add_classic_graph():
    x = torch.randn(4, 8).to("nntile").requires_grad_(True)
    y = relu(x)
    ones = torch.ones(tuple(y.shape), dtype=y.dtype)
    torch.autograd.grad(y, x, grad_outputs=ones.to(y.device))
    assert_classic_graph()

    a = torch.randn(4, 8).to("nntile").requires_grad_(True)
    b = torch.randn(4, 8).to("nntile").requires_grad_(True)
    z = add(a, b)
    ones_z = torch.ones(tuple(z.shape), dtype=z.dtype)
    torch.autograd.grad(z, [a, b], grad_outputs=ones_z.to(z.device))
    assert_classic_graph()


def test_fan_in_backward_classic_graph():
    """Same activation used twice: autograd combine must be classic ADD."""
    x = torch.randn(4, 8).to("nntile").requires_grad_(True)
    y = add(x, relu(x))
    ones = torch.ones(tuple(y.shape), dtype=y.dtype)
    torch.autograd.grad(y, x, grad_outputs=ones.to(y.device))
    assert_classic_graph()


def test_gelu_forward_matches_cpu():
    x = torch.randn(4, 8)
    y_cpu = F.gelu(x, approximate="tanh")
    y_n = nntile_cpu(gelu(x.to("nntile"), approximate="tanh"))
    assert torch.allclose(y_n, y_cpu, rtol=1e-4, atol=1e-4)


def _cyclic_shift(t: torch.Tensor, rot: int) -> torch.Tensor:
    n = t.dim()
    return t.permute(*[(i + rot) % n for i in range(n)]).contiguous()


def test_model_transpose_nonleaf_backward():
    """dX = dY.T(); non-leaf input is not saved for backward."""
    torch.manual_seed(3)
    x = torch.randn(2, 3, 4, 8, requires_grad=True)
    model_ndim = 1
    h_cpu = F.relu(x)
    y_cpu = _cyclic_shift(h_cpu, h_cpu.dim() - model_ndim)
    dy = torch.randn_like(y_cpu)
    (gx_cpu,) = torch.autograd.grad(y_cpu, x, grad_outputs=dy)

    x_n = x.detach().to("nntile").requires_grad_(True)
    h_n = relu(x_n)
    y_n = nntile_model_transpose(h_n, model_ndim)
    saved = getattr(y_n.grad_fn, "saved_tensors", None)
    if saved is not None:
        assert tuple(saved) == ()
    (gx_n,) = torch.autograd.grad(
        y_n,
        x_n,
        grad_outputs=dy.contiguous().to("nntile"),
    )
    assert torch.allclose(nntile_cpu(gx_n), gx_cpu, rtol=1e-4, atol=1e-4)
    assert_classic_graph()


def test_relu_saves_output_not_input():
    """Like torch.nn ReluBackward0: only the output is saved (mask)."""
    torch.manual_seed(5)
    x = torch.randn(4, 8, requires_grad=True)
    packed_cpu = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: packed_cpu.append(t) or t,
        lambda t: t,
    ):
        y_cpu = F.relu(x)
    assert packed_cpu == [y_cpu]
    dy = torch.randn_like(y_cpu)
    (gx_cpu,) = torch.autograd.grad(y_cpu, x, grad_outputs=dy)

    packed = []
    x_n = x.detach().to("nntile").requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: packed.append(t) or t,
        lambda t: t,
    ):
        y_n = relu(x_n)
    assert packed == [y_n]
    (gx_n,) = torch.autograd.grad(
        y_n,
        x_n,
        grad_outputs=dy.contiguous().to("nntile"),
    )
    assert torch.allclose(nntile_cpu(gx_n), gx_cpu, rtol=1e-4, atol=1e-4)
    assert_classic_graph()


def test_add_nonleaf_backward():
    """d(x+y) = dZ; operands are not saved for backward."""
    torch.manual_seed(4)
    x = torch.randn(4, 8, requires_grad=True)
    y = torch.randn(4, 8, requires_grad=True)
    hx_cpu = F.relu(x)
    hy_cpu = F.relu(y)
    z_cpu = hx_cpu + hy_cpu
    dz = torch.randn_like(z_cpu)
    gx_cpu, gy_cpu = torch.autograd.grad(z_cpu, (x, y), grad_outputs=dz)

    x_n = x.detach().to("nntile").requires_grad_(True)
    y_n = y.detach().to("nntile").requires_grad_(True)
    z_n = add(relu(x_n), relu(y_n))
    saved = getattr(z_n.grad_fn, "saved_tensors", None)
    if saved is not None:
        assert tuple(saved) == ()
    gx_n, gy_n = torch.autograd.grad(
        z_n,
        [x_n, y_n],
        grad_outputs=dz.contiguous().to("nntile"),
    )
    assert torch.allclose(nntile_cpu(gx_n), gx_cpu, rtol=1e-4, atol=1e-4)
    assert torch.allclose(nntile_cpu(gy_n), gy_cpu, rtol=1e-4, atol=1e-4)
    assert_classic_graph()

