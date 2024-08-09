# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_conv2d.py
# Test for tensor::conv2d<T> Python wrapper
#
# @version 1.1.0

from typing import Sequence

import numpy as np
import pytest
import torch

import nntile
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
        'fp32_fast_tf32': {'rtol': 8e-4},
        'bf16': {'rtol': 1.6e-2},
}

dtype2tol_weight = {
        'fp32': {'rtol': 5e-5},
        'fp32_fast_tf32': {'rtol': 8e-4},
        'bf16': {'rtol': 1.6e-2},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


def generate_inputs(numpy_rng, dtype: str, in_channels: int, out_channels: int,
        kernel: Sequence[int], H_in: int, H_in_tile: int, W_in: int,
        W_in_tile: int, batch: int, batch_tile: int, padding: Sequence[int],
        stride: Sequence[int], dilation: Sequence[int]):
    torch_layer = torch.nn.Conv2d(in_channels, out_channels,
            kernel_size=kernel, padding=padding, stride=stride,
            dilation=dilation, bias=False)
    x_shape = [W_in, H_in, in_channels, batch]
    x_basetile = [W_in_tile, H_in_tile, in_channels, batch_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = dtype2nntile[dtype]
    x_value = x_type(x_traits, x_distr, 0)
    x_grad = x_type(x_traits, x_distr, 0)
    X = TensorMoments(x_value, x_grad, grad_required=True)
    x_random = numpy_rng.standard_normal(x_shape, dtype=np.float32)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.tensor(x_nntile.T, requires_grad=True)

    nntile_layer, _ = nntile.layer.Conv2d.from_torch(torch_layer, X, 0)
    y_grad_random = numpy_rng.standard_normal(nntile_layer.y.value.shape,
            dtype=np.float32)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_layer.y.grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    return torch_layer, nntile_layer, x_torch, y_grad_torch


@pytest.mark.parametrize('dtype', [
    'fp32',
    'fp32_fast_tf32',
    pytest.param('bf16', marks=nocuda)
])
@pytest.mark.parametrize('in_channels,out_channels', [[2, 4]])
@pytest.mark.parametrize('kernel', [[3, 2]])
@pytest.mark.parametrize('H_in,H_in_tile', [[7, 2]])
@pytest.mark.parametrize('W_in,W_in_tile', [[6, 2]])
@pytest.mark.parametrize('batch,batch_tile', [[3, 2]])
@pytest.mark.parametrize('padding', [[2, 4]])
@pytest.mark.parametrize('stride', [[2, 3]])
@pytest.mark.parametrize('dilation', [[3, 2]])
def test_coercion(starpu_simple, numpy_rng, dtype: str,
        in_channels: int, out_channels: int, kernel: Sequence[int],
        H_in: int, H_in_tile: int, W_in: int, W_in_tile: int, batch: int,
        batch_tile: int, padding: Sequence[int], stride: Sequence[int],
        dilation: Sequence[int]):
    # Run tests only if output shape is positive
    H_out = H_in + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1
    W_out = W_in + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1
    if H_out <= 0 or W_out <= 0:
        return
    torch_layer, nntile_layer, *_ = generate_inputs(numpy_rng, dtype,
            in_channels, out_channels, kernel, H_in, H_in_tile, W_in,
            W_in_tile, batch, batch_tile, padding, stride, dilation)
    torch_layer_other = nntile_layer.to_torch()
    nntile_layer.unregister()
    nntile_layer.x.unregister()
    nntile_layer.y.unregister()

    assert torch_layer.in_channels == torch_layer_other.in_channels
    assert torch_layer.out_channels == torch_layer_other.out_channels
    assert torch_layer.kernel_size == torch_layer_other.kernel_size
    assert torch_layer.padding == torch_layer_other.padding
    assert torch_layer.stride == torch_layer_other.stride
    assert torch_layer.dilation == torch_layer_other.dilation

    rtol = dtype2tol[dtype]['rtol']
    for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
            torch_layer_other.named_parameters()):
        assert n1 == n2
        assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)


@pytest.mark.parametrize('dtype', [
    'fp32',
    # 'fp32_fast_tf32', # These tests are disabled because we need to test only
    #                   # small subset of all tests with these precisions
    # 'bf16'
])
@pytest.mark.parametrize('in_channels,out_channels', [[2, 4]])
@pytest.mark.parametrize('kernel', [
    [1, 1], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7], [9, 8]
])
@pytest.mark.parametrize('H_in,H_in_tile,W_in,W_in_tile', [
    [7, 1, 6, 99], [7, 2, 6, 1], [7, 3, 6, 2], [7, 4, 6, 3], [7, 5, 6, 4],
    [7, 6, 6, 5], [7, 7, 6, 6], [7, 8, 6, 7], [7, 9, 6, 8], [7, 10, 6, 9],
    [7, 11, 6, 10], [7, 12, 6, 11]
])
@pytest.mark.parametrize('batch,batch_tile', [[3, 2]])
@pytest.mark.parametrize('padding', [[3, 4]])
@pytest.mark.parametrize('stride', [[2, 3]])
@pytest.mark.parametrize('dilation', [[3, 2]])
class TestConv2d:

    def test_forward(self, starpu_simple, numpy_rng, dtype: str,
            in_channels: int, out_channels: int, kernel: Sequence[int],
            H_in: int, H_in_tile: int, W_in: int, W_in_tile: int, batch: int,
            batch_tile: int, padding: Sequence[int], stride: Sequence[int],
            dilation: Sequence[int]):
        # Run tests only if output shape is positive
        H_out = H_in + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1
        W_out = W_in + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1
        if H_out <= 0 or W_out <= 0:
            return
        torch_layer, nntile_layer, x, _ = generate_inputs(
                numpy_rng, dtype, in_channels, out_channels, kernel, H_in,
                H_in_tile, W_in, W_in_tile, batch, batch_tile, padding, stride,
                dilation)
        y = torch_layer(x)
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.y.value).T)
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    def test_backward(self, starpu_simple, numpy_rng, dtype: str,
            in_channels: int, out_channels: int, kernel: Sequence[int],
            H_in: int, H_in_tile: int, W_in: int, W_in_tile: int, batch: int,
            batch_tile: int, padding: Sequence[int], stride: Sequence[int],
            dilation: Sequence[int]):
        # Run tests only if output shape is positive
        H_out = H_in + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1
        W_out = W_in + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1
        if H_out <= 0 or W_out <= 0:
            return
        torch_layer, nntile_layer, x, y_grad = generate_inputs(
                numpy_rng, dtype, in_channels, out_channels, kernel, H_in,
                H_in_tile, W_in, W_in_tile, batch, batch_tile, padding, stride,
                dilation)
        y = torch_layer(x)
        nntile_layer.forward_async()
        res = (y * y_grad).sum()
        res.backward()
        nntile_layer.backward_async()
        x_grad_nntile = torch.Tensor(to_numpy(nntile_layer.x.grad).T)
        torch_layer_other = nntile_layer.to_torch_with_grads()
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(x.grad - x_grad_nntile) <= rtol * torch.norm(x.grad)

        rtol = dtype2tol_weight[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                torch_layer_other.named_parameters()):
            assert n1 == n2
            assert p1.requires_grad == p2.requires_grad
            if p1.requires_grad:
                g1, g2 = p1.grad, p2.grad
                assert torch.norm(g1 - g2) <= rtol * torch.norm(g1)
