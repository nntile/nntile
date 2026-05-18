# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_mlp_mixer.py
# Test for nntile.model.MlpMixer
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch

import nntile
from nntile.model.mixer_block import (
    _copy_param_from_torch, _weight_layout_transposed,
)
from nntile.model.mlp_mixer import MlpMixer
from nntile.model.mlp_mixer_config import MlpMixerConfig
from nntile.tensor import TensorMoments, TensorTraits
from nntile.torch_models.mlp_mixer import MlpMixer as TorchMlpMixer
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
    'fp32': {'rtol': 1e-6, 'atol': 1e-5},
    'bf16': {'rtol': 4e-2, 'atol': 2e-2},
}

nocuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='no cuda'
)


@dataclass
class MlpMixerTestParams:
    n_patches: int
    n_patches_tile: int
    n_batch: int
    n_batch_tile: int
    init_patch_dim: int
    init_patch_tile: int
    projected_patch_dim: int
    num_mixer_layers: int
    n_classes: int


single_tile = MlpMixerTestParams(
    n_patches=8, n_patches_tile=8,
    n_batch=2, n_batch_tile=2,
    init_patch_dim=4, init_patch_tile=4,
    projected_patch_dim=4,
    num_mixer_layers=2, n_classes=3,
)

multiple_tiles = MlpMixerTestParams(
    n_patches=16, n_patches_tile=4,
    n_batch=3, n_batch_tile=1,
    init_patch_dim=8, init_patch_tile=4,
    projected_patch_dim=8,
    num_mixer_layers=2, n_classes=5,
)


def _assert_close(ref: torch.Tensor, val: torch.Tensor, dtype: str) -> None:
    tol = dtype2tol[dtype]
    assert torch.norm(ref - val) <= (
        tol['rtol'] * torch.norm(ref) + tol.get('atol', 0.0)
    )


def _grad_to_torch(grad_np: np.ndarray, torch_grad: torch.Tensor) -> torch.Tensor:
    grad = torch.tensor(grad_np)
    if _weight_layout_transposed(grad_np.shape, tuple(torch_grad.shape)):
        return grad.T
    return grad


def generate_inputs(dtype: str, params: MlpMixerTestParams):
    rng = np.random.default_rng(42)
    config = MlpMixerConfig(
        channel_dim=params.n_patches,
        init_patch_dim=params.init_patch_dim,
        projected_patch_dim=params.projected_patch_dim,
        num_mixer_layers=params.num_mixer_layers,
        n_classes=params.n_classes,
        dtype=dtype,
    )
    x_shape = [params.n_patches, params.n_batch, params.init_patch_dim]
    x_basetile = [
        params.n_patches_tile, params.n_batch_tile, params.init_patch_tile,
    ]
    x_type = dtype2nntile[dtype]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_value = x_type(x_traits, x_distr)
    x_grad = x_type(x_traits, x_distr)
    X = TensorMoments(x_value, x_grad, grad_required=True)

    x_np = np.array(
        rng.standard_normal(x_shape), dtype=dtype2np[dtype], order="F",
    )
    x_value.from_array(x_np)
    x_torch = torch.tensor(x_np, requires_grad=True)

    torch_model = TorchMlpMixer(
        params.n_patches, params.init_patch_dim,
        params.projected_patch_dim, params.num_mixer_layers, params.n_classes,
    )
    with torch.no_grad():
        for p in torch_model.parameters():
            p.copy_(torch.tensor(
                rng.standard_normal(p.shape), dtype=torch.float32,
            ))

    model = MlpMixer.generate_simple(X, config)
    for p_nntile, p_torch in zip(
        model.parameters, torch_model.parameters(),
    ):
        _copy_param_from_torch(p_torch, p_nntile)
    model.clear_gradients()

    y_grad_np = np.array(
        rng.standard_normal(model.activations[-1].value.shape),
        dtype=dtype2np[dtype], order="F",
    )
    model.activations[-1].grad.from_array(y_grad_np)
    y_grad_torch = torch.tensor(y_grad_np.T)

    return torch_model, model, x_torch, y_grad_torch


@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
@pytest.mark.parametrize('dtype', [
    'fp32',
    pytest.param('bf16', marks=nocuda),
])
class TestMlpMixer:

    def test_coercion(
        self, context, torch_rng, dtype: str, params: MlpMixerTestParams,
    ):
        torch_model, model, _, _ = generate_inputs(dtype, params)
        torch_other = model.to_torch()
        model.unregister()

        for (n1, p1), (n2, p2) in zip(
            torch_model.named_parameters(), torch_other.named_parameters(),
        ):
            assert n1 == n2
            _assert_close(p1, p2, dtype)

    def test_forward(
        self, context, torch_rng, dtype: str, params: MlpMixerTestParams,
    ):
        torch_model, model, x_torch, _ = generate_inputs(dtype, params)
        y_torch = torch_model(x_torch)
        model.forward_async()
        nntile.starpu.wait_for_all()
        y_nntile = torch.tensor(to_numpy(model.activations[-1].value).T)
        model.unregister()
        _assert_close(y_torch, y_nntile, dtype)

    def test_backward(
        self, context, torch_rng, dtype: str, params: MlpMixerTestParams,
    ):
        torch_model, model, x_torch, y_grad_torch = generate_inputs(
            dtype, params,
        )
        (torch_model(x_torch) * y_grad_torch).sum().backward()
        model.forward_async()
        model.backward_async()
        nntile.starpu.wait_for_all()

        _assert_close(x_torch.grad, torch.tensor(
            to_numpy(model.activations[0].grad),
        ), dtype)
        for p_nntile, p_torch in zip(
            model.parameters, torch_model.parameters()
        ):
            grad_np = np.zeros(
                p_nntile.grad.shape, dtype=dtype2np[dtype], order="F",
            )
            p_nntile.grad.to_array(grad_np)
            _assert_close(
                p_torch.grad, _grad_to_torch(grad_np, p_torch.grad), dtype
            )
        model.unregister()


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['bf16', 'fp16', 'fp32'])
def test_bench_mlp_mixer_forward_async(
        context_cuda, benchmark_operation, dtype: str,
):
    _, model, _, _ = generate_inputs(dtype, single_tile)

    def bench_fn():
        model.forward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)
    model.unregister()


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['bf16', 'fp16', 'fp32'])
def test_bench_mlp_mixer_forward_backward_async(
        context_cuda, benchmark_operation, dtype: str,
):
    _, model, _, _ = generate_inputs(dtype, single_tile)
    nntile.tensor.clear_async(model.activations[0].grad)
    for p in model.parameters:
        if p.grad is not None:
            nntile.tensor.clear_async(p.grad)
    model.forward_async()
    nntile.starpu.wait_for_all()

    def bench_fn():
        model.forward_async()
        model.backward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)
    model.unregister()
