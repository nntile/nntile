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
# @version 1.0.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn as nn

import nntile
import nntile.utils
import nntile.utils.constructors
from nntile.layer import BatchNorm2d


@dataclass
class BatchNormTestParams:
    shape: tuple
    dtype: np.dtype

    atol: float = 1e-05

    eps: float = 1e-05
    redux: bool = False


BATCH_NORM_2D_TEST_PARAMS = [
    BatchNormTestParams((4, 5, 10, 10), np.float32),
    BatchNormTestParams((4, 5, 10, 10), np.float64),
    BatchNormTestParams((1, 1, 10, 10), np.float32),
]


def generate_input(params: BatchNormTestParams):
    input_np = np.random.rand(*params.shape).astype(params.dtype)
    output_grad_np = np.random.rand(*params.shape).astype(params.dtype)

    input_torch = torch.tensor(input_np, requires_grad=True)
    output_grad_torch = torch.tensor(output_grad_np, requires_grad=False)

    input_nnt = nntile.tensor.astensor(input_np)
    input_grad_nnt = nntile.tensor.astensor(np.zeros(params.shape).astype(params.dtype))
    input_moment = nntile.tensor.TensorMoments(
        input_nnt, input_grad_nnt, grad_required=True
    )

    output_grad_nnt = nntile.tensor.astensor(output_grad_np)

    return (input_moment, output_grad_nnt), (input_torch, output_grad_torch)


@pytest.mark.parametrize("params", BATCH_NORM_2D_TEST_PARAMS)
class TestBatchNorm2d:
    def test_batchnorm_forward(self, starpu_simple, params: BatchNormTestParams):
        (input_moment, _), (input_torch, _) = generate_input(params)

        # torch forward
        out_torch = nn.BatchNorm2d(
            params.shape[1], dtype=input_torch.dtype, affine=False
        )(input_torch)

        # nntile forward
        nntile_layer = BatchNorm2d.generate_simple(
            input_moment, eps=params.eps, redux=params.redux
        )

        nntile_layer.forward_async()

        # test forward
        np.testing.assert_allclose(
            nntile.tensor.asarray(nntile_layer.x_res),
            out_torch.detach().numpy(),
            atol=params.atol,
            err_msg=f"Error in forward for params: {params}",
        )

    def test_batchnorm_backward(self, starpu_simple, params: BatchNormTestParams):
        (input_moment, output_grad_nnt), (input_torch, output_grad_torch) = (
            generate_input(params)
        )

        # torch forward/backward
        out_torch = nn.BatchNorm2d(
            params.shape[1], dtype=input_torch.dtype, affine=False
        )(input_torch)
        l = (out_torch * output_grad_torch.detach()).sum()
        l.backward()

        # nntile forward/backward
        nntile_layer = BatchNorm2d.generate_simple(
            input_moment, eps=params.eps, redux=params.redux
        )

        nntile_layer.forward_async()
        nntile_layer.grad = output_grad_nnt
        nntile_layer.backward_async()

        # test backward
        np.testing.assert_allclose(
            nntile.tensor.asarray(nntile_layer.x_tm.grad),
            input_torch.grad.numpy(),
            atol=params.atol,
            err_msg=f"Error in backward for params: {params}",
        )
