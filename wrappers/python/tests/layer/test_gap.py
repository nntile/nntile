# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_gap.py
# Test for nntile.layer.gap
#
# @version 1.1.0

import numpy as np
import pytest
import torch

import nntile
from nntile.layer import GAP as Gap_Layer

# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
        np.float64: nntile.tensor.Tensor_fp64}


dtype2nntile = {
    'fp16': nntile.tensor.Tensor_fp16,
    'bf16': nntile.tensor.Tensor_bf16,
    'fp32': nntile.tensor.Tensor_fp32,
}

dtype2np = {
    'fp16': np.float32,
    'bf16': np.float32,
    'fp32': np.float32,
}

@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_gap(context, dtype: np.dtype):
    if dtype == np.float32:
        tol = 1e-5
    elif dtype == np.float64:
        tol = 1e-10
    # Describe single-tile tensor, located at node 0
    A_shape = [8, 2, 4]
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    mpi_distr = [0]

    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr)
    A_grad = Tensor[dtype](A_traits, mpi_distr)

    # Set initial values of tensors
    rand_A = np.random.default_rng(42).standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)

    # Define mlp_mixer layer
    layer = Gap_Layer.generate_simple(A_moments)

    A.from_array(np_A)
    layer.forward_async()

    torch_data = torch.from_numpy(np_A)
    torch_output = torch_data.mean(dim=(0))
    np_Y = np.array(torch_output.detach().numpy(), order="F", dtype=dtype)
    np_Y = np_Y.transpose()

    np_Y2 = np.zeros_like(np_Y, order='F')
    layer.y.value.to_array(np_Y2)
    assert np.linalg.norm(np_Y - np_Y2) / np.linalg.norm(np_Y) <= tol
    A_moments.unregister()
    layer.unregister()


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['fp32', 'fp16', 'bf16'])
def test_bench_gap_forward_async(context_cuda, benchmark_operation, dtype: str):
    A_shape = [128, 64, 16]
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    mpi_distr = [0]

    tensor_type = dtype2nntile[dtype]
    A = tensor_type(A_traits, mpi_distr)
    A_grad = tensor_type(A_traits, mpi_distr)

    rng = np.random.default_rng(42)
    np_dtype = dtype2np[dtype]
    np_A = np.array(rng.standard_normal(A_shape), dtype=np_dtype, order='F')
    A.from_array(np_A)
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)

    layer = Gap_Layer.generate_simple(A_moments)

    out_np = np.zeros(layer.y.value.shape, dtype=np_dtype, order='F')

    def bench_fn():
        layer.forward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)

@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['fp32', 'fp16', 'bf16'])
def test_bench_gap_forward_backward_async(context_cuda, benchmark_operation, dtype: str):
    A_shape = [128, 64, 16]
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    mpi_distr = [0]

    tensor_type = dtype2nntile[dtype]
    A = tensor_type(A_traits, mpi_distr)
    A_grad = tensor_type(A_traits, mpi_distr)

    rng = np.random.default_rng(42)
    np_dtype = dtype2np[dtype]
    np_A = np.array(rng.standard_normal(A_shape), dtype=np_dtype, order='F')
    A.from_array(np_A)
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)

    layer = Gap_Layer.generate_simple(A_moments)

    nntile.tensor.clear_async(A_grad)
    layer.clear_gradients()
    # forward and set grad for y
    layer.forward_async()
    grad_np = np.array(rng.standard_normal(layer.y.value.shape), dtype=np_dtype, order='F')
    layer.y.grad.from_array(grad_np)

    def bench_fn():
        layer.forward_async()
        layer.backward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)
