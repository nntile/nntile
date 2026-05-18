# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_mixer_block.py
# Test for nntile.model.MixerBlock
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch

import nntile
from nntile.model.mixer_block import MixerBlock
from nntile.model.mlp_mixer_config import MlpMixerConfig
from nntile.tensor import TensorMoments, TensorTraits
from nntile.torch_models.mlp_mixer import Mixer as TorchMixer
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
class MixerBlockTestParams:
    n_patches: int
    n_patches_tile: int
    n_batch: int
    n_batch_tile: int
    n_channels: int
    n_channels_tile: int


single_tile = MixerBlockTestParams(
    n_patches=8,
    n_patches_tile=8,
    n_batch=2,
    n_batch_tile=2,
    n_channels=4,
    n_channels_tile=4,
)

multiple_tiles = MixerBlockTestParams(
    n_patches=16,
    n_patches_tile=4,
    n_batch=3,
    n_batch_tile=1,
    n_channels=8,
    n_channels_tile=4,
)


def _rand_array(rng, shape, dtype: str) -> np.ndarray:
    return np.array(
        rng.standard_normal(shape),
        dtype=dtype2np[dtype],
        order="F",
    )


def _make_config(params: MixerBlockTestParams, dtype: str) -> MlpMixerConfig:
    return MlpMixerConfig(
        channel_dim=params.n_patches,
        init_patch_dim=params.n_channels,
        projected_patch_dim=params.n_channels,
        num_mixer_layers=1,
        n_classes=1,
        dtype=dtype,
    )


def _param_grad_to_torch(
    p_nntile_grad: np.ndarray, p_torch_grad: torch.Tensor,
) -> torch.Tensor:
    grad_nntile = torch.tensor(p_nntile_grad)
    if p_torch_grad.shape[0] != p_nntile_grad.shape[0]:
        return grad_nntile.T
    return grad_nntile


def _load_block_weights(block: MixerBlock, rng, dtype: str) -> tuple:
    np_w1 = _rand_array(rng, block.mlp_1.linear_1.w.value.shape, dtype)
    np_w2 = _rand_array(rng, block.mlp_1.linear_2.w.value.shape, dtype)
    np_w3 = _rand_array(rng, block.mlp_2.linear_1.w.value.shape, dtype)
    np_w4 = _rand_array(rng, block.mlp_2.linear_2.w.value.shape, dtype)
    block.mlp_1.linear_1.w.value.from_array(np_w1)
    block.mlp_1.linear_2.w.value.from_array(np_w2)
    block.mlp_2.linear_1.w.value.from_array(np_w3)
    block.mlp_2.linear_2.w.value.from_array(np_w4)

    np_gamma_1 = _rand_array(rng, block.norm_1.gamma.value.shape, dtype)
    np_beta_1 = _rand_array(rng, block.norm_1.beta.value.shape, dtype)
    np_gamma_2 = _rand_array(rng, block.norm_2.gamma.value.shape, dtype)
    np_beta_2 = _rand_array(rng, block.norm_2.beta.value.shape, dtype)
    block.norm_1.gamma.value.from_array(np_gamma_1)
    block.norm_1.beta.value.from_array(np_beta_1)
    block.norm_2.gamma.value.from_array(np_gamma_2)
    block.norm_2.beta.value.from_array(np_beta_2)
    return (np_w1, np_w2, np_w3, np_w4, np_gamma_1, np_beta_1,
            np_gamma_2, np_beta_2)


def generate_inputs(dtype: str, params: MixerBlockTestParams):
    rng = np.random.default_rng(42)
    config = _make_config(params, dtype)
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

    x_nntile = _rand_array(rng, x_shape, dtype)
    x_value.from_array(x_nntile)
    x_torch = torch.tensor(x_nntile, requires_grad=True)

    block = MixerBlock.generate_simple(X, config)
    weights = _load_block_weights(block, rng, dtype)

    torch_layer = TorchMixer(params.n_patches, params.n_channels)
    torch_layer.set_weight_parameters(*weights[:4])
    torch_layer.set_normalization_parameters(*weights[4:])

    block.clear_gradients()
    y_grad_nntile = _rand_array(
        rng, block.activations[-1].value.shape, dtype,
    )
    block.activations[-1].grad.from_array(y_grad_nntile)
    y_grad_torch = torch.tensor(y_grad_nntile)

    return block, torch_layer, x_torch, y_grad_torch


def _unregister_block(block: MixerBlock) -> None:
    block.unregister()


@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
@pytest.mark.parametrize('dtype', [
    'fp32',
    pytest.param('bf16', marks=nocuda),
])
class TestMixerBlock:

    def test_torch_coercion(
        self, context, torch_rng, dtype: str,
        params: MixerBlockTestParams,
    ):
        block, torch_ref, _, _ = generate_inputs(dtype, params)
        torch_block = block.to_torch()
        _unregister_block(block)

        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(
            torch_ref.named_parameters(), torch_block.named_parameters(),
        ):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)

    def test_forward(
        self, context, torch_rng, dtype: str,
        params: MixerBlockTestParams,
    ):
        block, torch_layer, x_torch, _ = generate_inputs(dtype, params)
        y_torch = torch_layer(x_torch)
        block.forward_async()
        nntile.starpu.wait_for_all()
        y_nntile = torch.tensor(to_numpy(block.activations[-1].value))
        _unregister_block(block)

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y_torch - y_nntile) <= rtol * torch.norm(y_torch)

    def test_backward(
        self, context, torch_rng, dtype: str,
        params: MixerBlockTestParams,
    ):
        block, torch_layer, x_torch, y_grad_torch = generate_inputs(
            dtype, params,
        )
        y_torch = torch_layer(x_torch)
        res = (y_torch * y_grad_torch).sum()
        res.backward()

        block.forward_async()
        block.backward_async()
        nntile.starpu.wait_for_all()

        grad_nntile = torch.tensor(to_numpy(block.activations[0].grad))
        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(x_torch.grad - grad_nntile) <= (
            rtol * torch.norm(x_torch.grad)
        )

        for p_nntile, p_torch in zip(
            block.parameters, torch_layer.parameters(),
        ):
            assert p_torch.grad is not None
            grad_np = np.zeros(
                p_nntile.grad.shape, dtype=dtype2np[dtype], order="F",
            )
            p_nntile.grad.to_array(grad_np)
            grad_torch = _param_grad_to_torch(grad_np, p_torch.grad)
            assert torch.norm(p_torch.grad - grad_torch) <= (
                rtol * torch.norm(p_torch.grad)
            )

        _unregister_block(block)


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['bf16', 'fp16', 'fp32'])
def test_bench_mixer_block_forward_async(
        context_cuda, benchmark_operation, dtype: str,
):
    params = single_tile
    block, _, _, _ = generate_inputs(dtype, params)

    def bench_fn():
        block.forward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)
    _unregister_block(block)


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['bf16', 'fp16', 'fp32'])
def test_bench_mixer_block_forward_backward_async(
        context_cuda, benchmark_operation, dtype: str,
):
    params = single_tile
    block, _, _, _ = generate_inputs(dtype, params)
    nntile.tensor.clear_async(block.activations[0].grad)
    for p in block.parameters:
        if p.grad is not None:
            nntile.tensor.clear_async(p.grad)
    block.forward_async()
    nntile.starpu.wait_for_all()

    def bench_fn():
        block.forward_async()
        block.backward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)
    _unregister_block(block)
