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

import nntile
from nntile.layer import BatchNorm2d

import nntile.utils
import nntile.utils.constructors
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

config = nntile.starpu.Config(1, 0, 0)

nntile.starpu.init()
dtypes = [np.float32, np.float64]

@dataclass
class BatchNormTestParams:
    shape: list
    dtype: np.dtype

    atol: float = 1e-05

    eps: float = 1e-05
    redux: bool = False

def generate_input(params: BatchNormTestParams):
    input_np = np.random.rand(*params.shape).astype(params.dtype)
    output_grad_np = np.random.rand(*params.shape).astype(params.dtype)

    input_torch = torch.tensor(input_np, requires_grad=True)
    output_grad_torch = torch.tensor(output_grad_np, requires_grad=False)

    # input_np = np.asfortranarray(input_np)
    # output_grad_np = np.asfortranarray(output_grad_np)

    input_nnt = nntile.tensor.from_array(input_np)
    input_grad_nnt = nntile.tensor.from_array(np.zeros(params.shape).astype(params.dtype))
    input_moment = nntile.tensor.TensorMoments(input_nnt, input_grad_nnt, grad_required=True)
    
    output_grad_nnt = nntile.tensor.from_array(output_grad_np)

    return (input_moment, output_grad_nnt), (input_torch, output_grad_torch)


def test_batchnorm(params: BatchNormTestParams):
    (input_moment, output_grad_nnt), (input_torch, output_grad_torch) = generate_input(params)
    
    # torch forward/backward
    o = nn.BatchNorm2d(params.shape[1], dtype=input_torch.dtype, affine=False)(input_torch)
    l = (o*output_grad_torch.detach()).sum()
    l.backward()

    # nntile forward/backward
    nntile_layer = BatchNorm2d.generate_simple(input_moment, eps=params.eps, redux=params.redux)
    
    nntile_layer.forward_async()
    nntile_layer.grad = output_grad_nnt
    nntile_layer.backward_async()

    # test forward
    success = torch.allclose(
        torch.tensor(nntile.tensor.to_array(nntile_layer.x_res)),
        o,
        atol=params.atol
    )
    assert success, "Error in forward for params: " + str(params)

    # test backward
    success = torch.allclose(
        torch.tensor(nntile.tensor.to_array(nntile_layer.x_tm.grad)),
        input_torch.grad,
        atol=params.atol
    )
    assert success, "Error in backward for params: " + str(params)

def batch_norm_test_suite():
    for _iter in range(4):
        test_batchnorm(BatchNormTestParams([4,5,10,10], np.float32))
        test_batchnorm(BatchNormTestParams([4,5,10,10], np.float64))

        test_batchnorm(BatchNormTestParams([1,1,10,10], np.float32))

if __name__ == "__main__":
    batch_norm_test_suite()
