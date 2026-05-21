# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_mixer_mlp.py
# Test for nntile.layer.MixerMlp
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch

import nntile
from nntile.tensor import TensorMoments, TensorTraits
from nntile.torch_models.mlp_mixer import MixerMlp as TorchMixerMlp
from nntile.utils.constructors import to_numpy

dtype2nntile = {
    'fp32': nntile.tensor.Tensor_fp32,
    'fp16': nntile.tensor.Tensor_fp16,
    'bf16': nntile.tensor.Tensor_bf16,
}

dtype2np = {
    'fp16': np.float32,
    'bf16': np.float32,
    'fp32': np.float32,
}

dtype2tol = {
    'fp32': {'rtol': 1e-6},
    'bf16': {'rtol': 4e-2},
}

nocuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='no cuda'
)


@dataclass
class MixerMlpTestParams:
    n_patches: int
    n_patches_tile: int
    n_batch: int
    n_batch_tile: int
    n_channels: int
    n_channels_tile: int


single_tile = MixerMlpTestParams(
    n_patches=8,
    n_patches_tile=8,
    n_batch=2,
    n_batch_tile=2,
    n_channels=4,
    n_channels_tile=4,
)

multiple_tiles = MixerMlpTestParams(
    n_patches=16,
    n_patches_tile=4,
    n_batch=3,
    n_batch_tile=1,
    n_channels=8,
    n_channels_tile=4,
)


def _torch_mixer_dim(side: str, params: MixerMlpTestParams) -> int:
    if side == 'L':
        return params.n_channels
    if side == 'R':
        return params.n_patches
    raise ValueError("side must be either 'L' or 'R'")


def _param_grad_torch(side: str, p_nntile_grad: np.ndarray) -> torch.Tensor:
    grad_nntile = torch.tensor(p_nntile_grad)
    if side == 'L':
        return grad_nntile.T
    return grad_nntile


def generate_inputs(
    dtype: str, params: MixerMlpTestParams, side: str,
):
    rng = np.random.default_rng(42)
    x_shape = [
        params.n_patches, params.n_batch, params.n_channels,
    ]
    x_basetile = [
        params.n_patches_tile, params.n_batch_tile, params.n_channels_tile,
    ]
    x_type = dtype2nntile[dtype]

    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_value = x_type(x_traits, x_distr)
    x_grad = x_type(x_traits, x_distr)
    X = TensorMoments(x_value, x_grad, grad_required=True)

    x_random = rng.standard_normal(x_shape)
    x_nntile = np.array(x_random, dtype=dtype2np[dtype], order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.tensor(x_nntile, requires_grad=True)

    nntile_layer = nntile.layer.MixerMlp.generate_simple(X, side)
    np_w1 = np.array(
        rng.standard_normal(nntile_layer.linear_1.w.value.shape),
        dtype=dtype2np[dtype],
        order="F",
    )
    np_w2 = np.array(
        rng.standard_normal(nntile_layer.linear_2.w.value.shape),
        dtype=dtype2np[dtype],
        order="F",
    )
    nntile_layer.linear_1.w.value.from_array(np_w1)
    nntile_layer.linear_2.w.value.from_array(np_w2)

    torch_layer = TorchMixerMlp(side, _torch_mixer_dim(side, params))
    torch_layer.set_weight(np_w1, np_w2)

    nntile_layer.clear_gradients()

    y_grad_random = rng.standard_normal(nntile_layer.y.value.shape)
    y_grad_nntile = np.array(
        y_grad_random, dtype=dtype2np[dtype], order="F",
    )
    nntile_layer.y.grad.from_array(y_grad_nntile)
    y_grad_torch = torch.tensor(y_grad_nntile)

    return nntile_layer, torch_layer, x_torch, y_grad_torch


def _unregister_layer(layer: nntile.layer.MixerMlp) -> None:
    layer.unregister()
    layer.x.unregister()
    layer.y.unregister()


@pytest.mark.parametrize('side', ['L', 'R'])
@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
@pytest.mark.parametrize('dtype', [
    'fp32',
    pytest.param('bf16', marks=nocuda),
])
class TestMixerMlp:

    def test_forward(
        self, context, torch_rng, side: str, dtype: str,
        params: MixerMlpTestParams,
    ):
        nntile_layer, torch_layer, x_torch, _ = generate_inputs(
            dtype, params, side,
        )
        y_torch = torch_layer(x_torch)
        nntile_layer.forward_async()
        nntile.starpu.wait_for_all()
        y_nntile = torch.tensor(to_numpy(nntile_layer.y.value))
        _unregister_layer(nntile_layer)

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y_torch - y_nntile) <= rtol * torch.norm(y_torch)

    def test_backward(
        self, context, torch_rng, side: str, dtype: str,
        params: MixerMlpTestParams,
    ):
        nntile_layer, torch_layer, x_torch, y_grad_torch = generate_inputs(
            dtype, params, side,
        )
        y_torch = torch_layer(x_torch)
        res = (y_torch * y_grad_torch).sum()
        res.backward()

        nntile_layer.forward_async()
        nntile_layer.backward_async()
        nntile.starpu.wait_for_all()

        grad_nntile = torch.tensor(to_numpy(nntile_layer.x.grad))
        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(x_torch.grad - grad_nntile) <= (
            rtol * torch.norm(x_torch.grad)
        )

        for p_nntile, p_torch in zip(
            nntile_layer.parameters, torch_layer.parameters(),
        ):
            assert p_torch.grad is not None
            grad_np = np.zeros(
                p_nntile.grad.shape, dtype=dtype2np[dtype], order="F",
            )
            p_nntile.grad.to_array(grad_np)
            grad_torch = _param_grad_torch(side, grad_np)
            assert torch.norm(p_torch.grad - grad_torch) <= (
                rtol * torch.norm(p_torch.grad)
            )

        _unregister_layer(nntile_layer)


@pytest.mark.benchmark
@pytest.mark.parametrize('side', ['L', 'R'])
@pytest.mark.parametrize('dtype', ['bf16', 'fp16', 'fp32'])
def test_bench_mixer_mlp_forward_async(
        context_cuda, benchmark_operation, side: str, dtype: str,
):
    params = single_tile
    nntile_layer, _, _, _ = generate_inputs(dtype, params, side)

    def bench_fn():
        nntile_layer.forward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)
    _unregister_layer(nntile_layer)


@pytest.mark.benchmark
@pytest.mark.parametrize('side', ['L', 'R'])
@pytest.mark.parametrize('dtype', ['bf16', 'fp16', 'fp32'])
def test_bench_mixer_mlp_forward_backward_async(
        context_cuda, benchmark_operation, side: str, dtype: str,
):
    params = single_tile
    nntile_layer, _, _, _ = generate_inputs(dtype, params, side)
    nntile.tensor.clear_async(nntile_layer.x.grad)
    for p in nntile_layer.parameters:
        if p.grad is not None:
            nntile.tensor.clear_async(p.grad)
    nntile_layer.forward_async()
    nntile.starpu.wait_for_all()

    def bench_fn():
        nntile_layer.forward_async()
        nntile_layer.backward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)
    _unregister_layer(nntile_layer)
