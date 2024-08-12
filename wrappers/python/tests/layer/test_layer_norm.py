# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_layer_norm.py
# Test for nntile.layer.LayerNorm
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from torch.nn import LayerNorm

import nntile
import nntile.tensor
import nntile.utils.constructors as nntc
from nntile.tensor import TensorMoments, TensorTraits
from nntile.utils.constructors import to_numpy


# NNTile dtype via corresponding Tensor type
dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'bf16': nntile.tensor.Tensor_bf16,
}

dtype2tol = {
        'fp32': {'rtol': 1e-5},
        'fp32_fast_tf32': {'rtol': 8e-4},
        'bf16': {'rtol': 1.6e-2},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


@dataclass
class LayerNormTestParams:
    n_size: int
    n_size_dyn: int
    n_size_tile: int
    m_size: int
    m_size_tile: int


single_tile = LayerNormTestParams(
    n_size=10,
    n_size_dyn=20,
    n_size_tile=10,
    m_size=30,
    m_size_tile=30,
)

multiple_tiles = LayerNormTestParams(
    n_size=10,
    n_size_dyn=20,
    n_size_tile=5,
    m_size=30,
    m_size_tile=15,
)


def generate_inputs(dtype: str, params: LayerNormTestParams):
    rng = np.random.default_rng(42)
    eps = 1e-05

    torch_layer = LayerNorm(params.m_size, eps=eps)
    rand_gamma = rng.standard_normal(params.m_size)
    np_gamma = np.array(rand_gamma, dtype=np.float32, order='F')
    rand_beta = rng.standard_normal(params.m_size)
    np_beta = np.array(rand_beta, dtype=np.float32, order='F')
    torch_layer.weight.data = torch.tensor(np_gamma)
    torch_layer.bias.data = torch.tensor(np_beta)

    x_shape = [params.n_size, params.m_size]
    x_basetile = [params.n_size_tile, params.m_size_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = dtype2nntile[dtype]
    x_value = x_type(x_traits, x_distr, 0)
    x_grad = x_type(x_traits, x_distr, 0)
    X = TensorMoments(x_value, x_grad, grad_required=True)
    x_random = rng.standard_normal(x_shape)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.Tensor(x_nntile)
    x_torch.requires_grad_()

    nntile_layer, _ = nntile.layer.LayerNorm.from_torch(torch_layer, X, 1, eps, 0)
    y_grad_random = rng.standard_normal(x_shape)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_layer.y.grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile)
    nntile.tensor.clear_async(nntile_layer.x.grad)
    nntile.tensor.clear_async(nntile_layer.gamma.grad)
    nntile.tensor.clear_async(nntile_layer.beta.grad)
    return torch_layer, nntile_layer, x_torch, y_grad_torch

def generate_inputs_dynamic(dtype: str, params: LayerNormTestParams):
    rng = np.random.default_rng(42)
    eps = 1e-05

    torch_layer = LayerNorm(params.m_size, eps=eps)
    rand_gamma = rng.standard_normal(params.m_size)
    np_gamma = np.array(rand_gamma, dtype=np.float32, order='F')
    rand_beta = rng.standard_normal(params.m_size)
    np_beta = np.array(rand_beta, dtype=np.float32, order='F')
    torch_layer.weight.data = torch.tensor(np_gamma)
    torch_layer.bias.data = torch.tensor(np_beta)

    x_shape = [params.n_size, params.m_size]
    x_basetile_shape = [params.n_size_tile, params.m_size_tile]
    x_random = rng.standard_normal(x_shape)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    X = nntile.tensor.TensorMoments(nntc.from_array(x_nntile, x_basetile_shape), None, False)
    x_torch = torch.Tensor(x_nntile)

    nntile_layer, _ = nntile.layer.LayerNorm.from_torch(torch_layer, X, 1, eps, 0)

    x_shape = [params.n_size_dyn, params.m_size]
    x_random = rng.standard_normal(x_shape)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    X_other = nntile.tensor.TensorMoments(nntc.from_array(x_nntile, x_basetile_shape), None, False)
    x_torch_other = torch.Tensor(x_nntile)
    return torch_layer, nntile_layer, x_torch, X, x_torch_other, X_other



@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
@pytest.mark.parametrize('dtype', [
    'fp32',
    pytest.param('fp32_fast_tf32', marks=nocuda),
    pytest.param('bf16', marks=nocuda),
])
class TestLayerNorm:

    def test_torch_coercion(self, starpu_simple, torch_rng, dtype: str,
                            params: LayerNormTestParams):
        torch_layer, nntile_layer, *_ = generate_inputs(dtype, params)
        torch_layer_other = nntile_layer.to_torch()
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                torch_layer_other.named_parameters()):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)

    def test_forward(self, starpu_simple, torch_rng, dtype: str,
                     params: LayerNormTestParams):
        torch_layer, nntile_layer, x, *_ = generate_inputs(dtype, params)
        y = torch_layer(x)
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.y.value))
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    def test_forward_dynamic(self, starpu_simple, torch_rng, dtype: str,
                             params: LayerNormTestParams):
        torch_layer, nntile_layer, x_torch, x_nntile, x_torch_other, x_nntile_other = \
            generate_inputs_dynamic(dtype, params)
        y = torch_layer(x_torch)
        outs_nnt = nntile_layer.forward_dynamic(x_nntile)
        y_nntile = torch.Tensor(nntc.to_numpy(outs_nnt.value))
        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

        y = torch_layer(x_torch_other)
        outs_nnt = nntile_layer.forward_dynamic(x_nntile_other)
        y_nntile = torch.Tensor(nntc.to_numpy(outs_nnt.value))
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    def test_backward(self, starpu_simple, torch_rng, dtype: str,
                              params: LayerNormTestParams):
        torch_layer, nntile_layer, x, y_grad = generate_inputs(dtype, params)
        y = torch_layer(x)
        nntile_layer.forward_async()
        res = (y * y_grad).sum()
        res.backward()
        nntile_layer.backward_async()
        torch_layer_other = nntile_layer.to_torch_with_grads()
        input_grad_nntile = torch.Tensor(to_numpy(nntile_layer.x.grad))
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(x.grad - input_grad_nntile) <= rtol * torch.norm(x.grad)
        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                torch_layer_other.named_parameters()):
            assert n1 == n2
            assert p1.requires_grad == p2.requires_grad
            if p1.requires_grad:
                g1, g2 = p1.grad, p2.grad
                assert torch.norm(g1 - g2) <= rtol * torch.norm(g1)
