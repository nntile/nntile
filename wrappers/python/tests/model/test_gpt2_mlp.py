# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_gpt2_mlp.py
# Test for nntile.layer.GPT2MLP
# Each test is generated in float precision by PyTorch, then it is downcasted
# into NNTile type. So, implementation of double precision is NOT checked.
#
# @version 1.0.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP, GPT2Config

import nntile
from nntile.model.gpt2_config import GPT2ConfigNNTile
from nntile.model.gpt2_mlp import GPT2MLP as GPT2MLP_nntile
from nntile.tensor import TensorMoments, TensorTraits
from nntile.utils.constructors import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'bf16': nntile.tensor.Tensor_bf16,
}

dtype2tol = {
        'fp32': {'rtol': 1e-6},
        'fp32_fast_tf32': {'rtol': 6e-4},
        'bf16': {'rtol': 1.6e-2},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


@dataclass
class GPT2MLPTestParams:
    hidden_size: int
    hidden_size_tile: int
    intermediate_size: int
    intermediate_size_tile: int
    n_batch: int
    n_batch_tile: int
    redux: bool = True
    seq_len: int = 100
    seq_len_tile: int = 100


multiple_tiles = GPT2MLPTestParams(
    hidden_size=128,
    hidden_size_tile=32,
    intermediate_size=64,
    intermediate_size_tile=16,
    seq_len=256,
    seq_len_tile=16,
    n_batch=4,
    n_batch_tile=1)

single_tile = GPT2MLPTestParams(
    hidden_size=128,
    hidden_size_tile=128,
    intermediate_size=64,
    intermediate_size_tile=64,
    seq_len=64,
    seq_len_tile=64,
    n_batch=3,
    n_batch_tile=3,
)


def generate_inputs(params: GPT2MLPTestParams, dtype: str):
    torch_layer_config = GPT2Config(
        n_embd=params.hidden_size,
        attn_pdrop = 0.0,
        resid_pdrop = 0.0,
        embd_pdrop = 0.0,
        use_cache=False,
    )
    torch_layer = GPT2MLP(
        params.intermediate_size,
        torch_layer_config
    )
    nntile_config = GPT2ConfigNNTile(
        vocab_size=torch_layer_config.vocab_size,
        vocab_embed_dim_tile=params.hidden_size,
        hidden_size=params.hidden_size,
        hidden_size_tile=params.hidden_size_tile,
        intermediate_size=params.intermediate_size,
        intermediate_size_tile=params.intermediate_size_tile,
        n_head=torch_layer_config.n_head,
        n_head_tile=torch_layer_config.n_head,
        dtype=dtype
    )
    x_shape = [params.hidden_size, params.seq_len, params.n_batch]
    x_basetile = [params.hidden_size_tile,
                  params.seq_len_tile,
                  params.n_batch_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = dtype2nntile[dtype]
    x_value = x_type(x_traits, x_distr, 0)
    x_grad = x_type(x_traits, x_distr, 0)
    X = TensorMoments(x_value, x_grad, grad_required=True)
    gen = np.random.default_rng(42)
    x_random = gen.standard_normal(x_shape, dtype=np.float32)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.Tensor(x_nntile.T)
    x_torch.requires_grad_()
    nntile_layer, _ = GPT2MLP_nntile.from_torch(torch_layer, X,
                                                nntile_config, 0)
    nntile_layer.clear_gradients()
    y_grad_random = gen.standard_normal(x_shape, dtype=np.float32)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_layer.activations[-1].grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    return torch_layer, nntile_layer, x_torch, y_grad_torch


@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
@pytest.mark.parametrize('dtype', [
    'fp32',
    pytest.param('fp32_fast_tf32', marks=nocuda),
    pytest.param('bf16', marks=nocuda),
])
class TestGPT2MLP:

    def test_coercion(self, starpu_simple, torch_rng,
                      params: GPT2MLPTestParams, dtype: str):
        torch_layer, nntile_layer, _, _ = generate_inputs(params, dtype)
        torch_layer_other = nntile_layer.to_torch()
        nntile_layer.unregister()

        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                torch_layer_other.named_parameters()):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)

    def test_forward(self, starpu_simple, torch_rng,
                     params: GPT2MLPTestParams,
                     dtype: str):
        torch_layer, nntile_layer, x, _ = generate_inputs(params, dtype)
        y = torch_layer(x)
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.activations[-1].value).T)
        nntile_layer.unregister()
        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    def test_backward(self, starpu_simple, torch_rng,
                              params: GPT2MLPTestParams,
                              dtype: str):
        torch_layer, nntile_layer, x, y_grad = generate_inputs(params, dtype)
        torch_layer_other = nntile_layer.to_torch()
        y = torch_layer(x)
        nntile_layer.forward_async()
        res = (y * y_grad).sum()
        res.backward()
        nntile_layer.backward_async()
        torch_layer_other = nntile_layer.to_torch_with_grads()
        grad_nntile = torch.Tensor(
            to_numpy(nntile_layer.activations[0].grad).T
        )
        nntile_layer.unregister()
        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(x.grad - grad_nntile) <= rtol * torch.norm(x.grad)

        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                torch_layer_other.named_parameters()):
            assert n1 == n2
            assert p1.requires_grad == p2.requires_grad
            if p1.requires_grad:
                g1, g2 = p1.grad, p2.grad
                assert torch.norm(g1 - g2) <= rtol * torch.norm(g1)
