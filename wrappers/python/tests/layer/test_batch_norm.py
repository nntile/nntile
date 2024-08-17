# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_batch_norm.py
# Test for nntile.layer.BatchNorm2d
#
# @version 1.1.0

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
import torch.nn as nn

import nntile
import nntile.utils
from nntile.layer import BatchNorm2d


@dataclass
class BatchNormTestParams:
    shape: tuple
    dtype: np.dtype

    atol: float = field(init=False)

    eps: float = 1e-05
    redux: bool = False

    def __post_init__(self):
        self.atol = 1e-5 if self.dtype is np.float64 else 1e-6


BATCH_NORM_2D_TEST_PARAMS = [
    BatchNormTestParams((4, 5, 10, 10), np.float32),
    BatchNormTestParams((4, 5, 10, 10), np.float64),
    BatchNormTestParams((1, 1, 10, 10), np.float32),
]


def generate_input(params: BatchNormTestParams, rng):
    input_np = rng.random(params.shape).astype(params.dtype)
    output_grad_np = rng.random(params.shape).astype(params.dtype)
    weights_np = rng.random(params.shape[1:2]).astype(params.dtype)
    bias_np = rng.random(params.shape[1:2]).astype(params.dtype)

    input_torch = torch.tensor(input_np, requires_grad=True)
    output_grad_torch = torch.tensor(output_grad_np, requires_grad=False)
    weights_torch = torch.tensor(weights_np)
    bias_torch = torch.tensor(bias_np)

    input_nnt = nntile.tensor.from_array(input_np)
    input_grad_nnt = nntile.tensor.from_array(
        np.zeros(params.shape).astype(params.dtype)
    )
    input_moment = nntile.tensor.TensorMoments(
        input_nnt, input_grad_nnt, grad_required=True
    )
    weights_nnt = nntile.tensor.TensorMoments(
        nntile.tensor.from_array(weights_np),
        nntile.tensor.from_array(
            np.zeros(weights_np.shape).astype(params.dtype)
        ),
        grad_required=True,
    )

    bias_nnt = nntile.tensor.TensorMoments(
        nntile.tensor.from_array(bias_np),
        nntile.tensor.from_array(
            np.zeros(weights_np.shape).astype(params.dtype)
        ),
        grad_required=True,
    )

    output_grad_nnt = nntile.tensor.from_array(output_grad_np)

    return (input_moment, output_grad_nnt, weights_nnt, bias_nnt), (
        input_torch,
        output_grad_torch,
        weights_torch,
        bias_torch,
    )


@pytest.mark.parametrize("params", BATCH_NORM_2D_TEST_PARAMS)
class TestBatchNorm2d:
    def test_batchnorm_forward(
        self, starpu_simple, numpy_rng, torch_rng, params: BatchNormTestParams
    ):
        (
            (input_moment, _, weights_nnt, bias_nnt),
            (input_torch, _, weights_torch, bias_torch),
        ) = generate_input(params, numpy_rng)

        # torch forward
        bn_torch = nn.BatchNorm2d(
            params.shape[1], dtype=input_torch.dtype, affine=False
        )
        bn_torch.weight = torch.nn.Parameter(weights_torch)
        bn_torch.bias = torch.nn.Parameter(bias_torch)
        out_torch = bn_torch(input_torch)

        # nntile forward
        nntile_layer = BatchNorm2d.generate_simple(
            input_moment, eps=params.eps, redux=params.redux
        )
        nntile_layer.weight = weights_nnt
        nntile_layer.bias = bias_nnt

        nntile_layer.forward_async()

        # test forward
        np.testing.assert_allclose(
            nntile.tensor.to_numpy(nntile_layer.y.value),
            out_torch.detach().numpy(),
            atol=params.atol,
            err_msg=f"Error in forward for params: {params}",
        )

    def test_batchnorm_backward(
        self, starpu_simple, numpy_rng, torch_rng, params: BatchNormTestParams
    ):
        (
            (input_moment, output_grad_nnt, weights_nnt, bias_nnt),
            (input_torch, output_grad_torch, weights_torch, bias_torch),
        ) = generate_input(params, numpy_rng)

        # torch forward/backward
        bn_torch = nn.BatchNorm2d(
            params.shape[1], dtype=input_torch.dtype, affine=False
        )
        bn_torch.weight = torch.nn.Parameter(weights_torch)
        bn_torch.bias = torch.nn.Parameter(bias_torch)
        out_torch = bn_torch(input_torch)
        loss = (out_torch * output_grad_torch.detach()).sum()
        loss.backward()

        # nntile forward/backward
        nntile_layer = BatchNorm2d.generate_simple(
            input_moment, eps=params.eps, redux=params.redux
        )
        nntile_layer.weight = weights_nnt
        nntile_layer.bias = bias_nnt

        nntile_layer.forward_async()
        nntile_layer.y.grad = output_grad_nnt
        nntile_layer.backward_async()

        # test d(batch_norm)/d(bias)
        np.testing.assert_allclose(
            nntile.tensor.to_numpy(nntile_layer.bias.grad),
            bn_torch.bias.grad.numpy(),
            atol=params.atol,
            err_msg=f"Error in backward d(bn)/d(b) for params: {params}",
        )

        # test d(batch_norm)/d(weight)
        np.testing.assert_allclose(
            nntile.tensor.to_numpy(nntile_layer.weight.grad),
            bn_torch.weight.grad.numpy(),
            atol=1e-5,
            rtol=1e-4,
            err_msg=f"Error in backward test d(bn)/d(w) for params: {params}",
        )

        # test d(batch_norm)/d(input)
        np.testing.assert_allclose(
            nntile.tensor.to_numpy(nntile_layer.x.grad),
            input_torch.grad.numpy(),
            atol=params.atol,
            err_msg=f"Error in backward d(bn)/d(i) for params: {params}",
        )
