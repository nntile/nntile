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
        'fp16': nntile.tensor.Tensor_fp16,
}

dtype2np = {
    'fp16': np.float32,
    'bf16': np.float32,
    'fp32': np.float32,
}

dtype2tol = {
        'fp32': {'rtol': 1e-6},
        'fp32_fast_tf32': {'rtol': 8e-4},
        'bf16': {'rtol': 1.6e-2},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


@dataclass
class LayerNormTestParams:
    eps: float
    n_size: int
    n_size_tile: int
    m_size: int
    m_size_tile: int
    m_size_dyn: int
    m_size_dyn_tile: int


single_tile = LayerNormTestParams(
    eps=1e-05,
    n_size=10,
    n_size_tile=10,
    m_size=30,
    m_size_tile=30,
    m_size_dyn=50,
    m_size_dyn_tile=50,
)

multiple_tiles = LayerNormTestParams(
    eps=100.0,
    n_size=10,
    n_size_tile=5,
    m_size=30,
    m_size_tile=15,
    m_size_dyn=60,
    m_size_dyn_tile=30,
)


def generate_inputs(dtype: str, params: LayerNormTestParams):
    rng = np.random.default_rng(42)
    eps = params.eps

    torch_layer = LayerNorm(params.n_size, eps=eps)
    rand_gamma = rng.standard_normal(params.n_size)
    np_gamma = np.array(rand_gamma, dtype=np.float32, order='F')
    rand_beta = rng.standard_normal(params.n_size)
    np_beta = np.array(rand_beta, dtype=np.float32, order='F')
    torch_layer.weight.data = torch.tensor(np_gamma)
    torch_layer.bias.data = torch.tensor(np_beta)

    x_shape = [params.n_size, params.m_size]
    x_basetile = [params.n_size_tile, params.m_size_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = dtype2nntile[dtype]
    x_value = x_type(x_traits, x_distr)
    x_grad = nntc.zeros_like(x_value)
    X = TensorMoments(x_value, x_grad, grad_required=True)
    x_random = rng.standard_normal(x_shape)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.Tensor(x_nntile.T)
    x_torch.requires_grad_()

    nntile_layer = nntile.layer.LayerNorm.from_torch(torch_layer, X)
    nntile_layer.clear_gradients()
    y_grad_random = rng.standard_normal(x_shape)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_layer.y.grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    nntile.tensor.clear_async(nntile_layer.gamma.grad)
    nntile.tensor.clear_async(nntile_layer.beta.grad)
    return torch_layer, nntile_layer, x_torch, y_grad_torch


def generate_inputs_dynamic(dtype: str, params: LayerNormTestParams):
    rng = np.random.default_rng(42)
    eps = params.eps

    torch_layer = LayerNorm(params.n_size, eps=eps)
    rand_gamma = rng.standard_normal(params.n_size)
    np_gamma = np.array(rand_gamma, dtype=np.float32, order='F')
    rand_beta = rng.standard_normal(params.n_size)
    np_beta = np.array(rand_beta, dtype=np.float32, order='F')
    torch_layer.weight.data = torch.tensor(np_gamma)
    torch_layer.bias.data = torch.tensor(np_beta)

    x_shape = [params.n_size, params.m_size]
    x_basetile_shape = [params.n_size_tile, params.m_size_tile]
    x_random = rng.standard_normal(x_shape)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    X = nntile.tensor.TensorMoments(
        nntc.from_array(x_nntile, x_basetile_shape), None, False
    )
    x_torch = torch.Tensor(x_nntile.T)

    nntile_layer = nntile.layer.LayerNorm.from_torch(torch_layer, X)

    x_shape = [params.n_size, params.m_size_dyn]
    x_basetile_shape = [params.n_size_tile, params.m_size_dyn_tile]
    x_random = rng.standard_normal(x_shape)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    X_other = nntile.tensor.TensorMoments(
        nntc.from_array(x_nntile, x_basetile_shape), None, False
    )
    x_torch_other = torch.Tensor(x_nntile.T)
    return torch_layer, nntile_layer, (x_torch, x_torch_other), (X, X_other)


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

    def test_torch_coercion(self, context, torch_rng, dtype: str,
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

    def test_forward(self, context, torch_rng, dtype: str,
                     params: LayerNormTestParams):
        torch_layer, nntile_layer, x, *_ = generate_inputs(dtype, params)
        y = torch_layer(x)
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.y.value).T)
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    def test_forward_dynamic(self, context, torch_rng, dtype: str,
                             params: LayerNormTestParams):
        torch_layer, nntile_layer, torch_inputs, nntile_inputs = \
            generate_inputs_dynamic(dtype, params)
        x_torch, x_torch_other = torch_inputs
        x_nntile, x_nntile_other = nntile_inputs
        y = torch_layer(x_torch)
        outs_nnt = nntile_layer.forward_dynamic(x_nntile)
        y_nntile = torch.Tensor(nntc.to_numpy(outs_nnt.value).T)
        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

        y = torch_layer(x_torch_other)
        outs_nnt = nntile_layer.forward_dynamic(x_nntile_other)
        y_nntile = torch.Tensor(nntc.to_numpy(outs_nnt.value).T)
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    def test_backward(self, context, torch_rng, dtype: str,
                              params: LayerNormTestParams):
        torch_layer, nntile_layer, x, y_grad = generate_inputs(dtype, params)
        y = torch_layer(x)
        nntile_layer.forward_async()
        res = (y * y_grad).sum()
        res.backward()
        nntile_layer.backward_async()
        torch_layer_other = nntile_layer.to_torch_with_grads()
        grad_nntile = torch.Tensor(to_numpy(nntile_layer.x.grad).T)
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(x.grad - grad_nntile) <= rtol * torch.norm(x.grad)
        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                torch_layer_other.named_parameters()):
            assert n1 == n2
            assert p1.requires_grad == p2.requires_grad
            if p1.requires_grad:
                g1, g2 = p1.grad, p2.grad
                assert torch.norm(g1 - g2) <= rtol * torch.norm(g1)


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['bf16', 'fp16', 'fp32'])
def test_bench_layernorm_forward_async(
        context_cuda, benchmark_operation, dtype: str,
):
    # minimal setup
    n_size = 128
    m_size = 256
    eps = 1e-5

    # torch layer to seed params
    torch_ln = LayerNorm(n_size, eps=eps)

    # build nntile input
    x_traits = TensorTraits([n_size, m_size], [n_size, m_size])
    x_distr = [0]
    x_type = dtype2nntile[dtype]
    x_val = x_type(x_traits, x_distr)
    x_grad = x_type(x_traits, x_distr)

    rng = np.random.default_rng(42)
    x_np = np.array(
        rng.standard_normal((n_size, m_size)),
        dtype=dtype2np[dtype],
        order="F",
    )
    x_val.from_array(x_np)
    nntile.tensor.clear_async(x_grad)

    X = TensorMoments(x_val, x_grad, grad_required=True)

    # build nntile layer from torch
    nnt_ln = nntile.layer.LayerNorm.from_torch(torch_ln, X)

    def bench_fn():
        nnt_ln.forward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['bf16', 'fp16', 'fp32'])
def test_bench_layernorm_forward_backward_async(
        context_cuda, benchmark_operation, dtype: str,
):
    n_size = 128
    m_size = 256
    eps = 1e-5

    torch_ln = LayerNorm(n_size, eps=eps)

    x_traits = TensorTraits([n_size, m_size], [n_size, m_size])
    x_distr = [0]
    x_type = dtype2nntile[dtype]
    x_val = x_type(x_traits, x_distr)
    x_grad = x_type(x_traits, x_distr)

    rng = np.random.default_rng(42)
    x_np = np.array(
        rng.standard_normal((n_size, m_size)),
        dtype=dtype2np[dtype],
        order="F",
    )
    x_val.from_array(x_np)

    X = TensorMoments(x_val, x_grad, grad_required=True)

    nnt_ln = nntile.layer.LayerNorm.from_torch(torch_ln, X)

    nnt_ln.clear_gradients()
    # prepare grad buffer
    grad_np = np.array(
        rng.standard_normal((n_size, m_size)),
        dtype=dtype2np[dtype],
        order="F",
    )

    def bench_fn():
        nnt_ln.forward_async()
        nnt_ln.y.grad.from_array(grad_np)
        nnt_ln.backward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)
