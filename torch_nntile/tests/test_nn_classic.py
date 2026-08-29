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


def test_gelu_forward_matches_cpu():
    x = torch.randn(4, 8)
    y_cpu = F.gelu(x, approximate="tanh")
    y_n = nntile_cpu(gelu(x.to("nntile"), approximate="tanh"))
    assert torch.allclose(y_n, y_cpu, rtol=1e-4, atol=1e-4)
