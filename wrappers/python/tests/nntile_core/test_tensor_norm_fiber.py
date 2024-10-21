# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_norm_fiber.py
# Test for tensor::add_scalar<T> Python wrapper
#
# @version 1.1.0

import numpy as np
import pytest

import nntile

config = nntile.starpu.Config(1, 0, 0)
nntile.starpu.init()

# Define mapping between numpy and nntile types
Tensor = {
    np.float32: nntile.tensor.Tensor_fp32,
    np.float64: nntile.tensor.Tensor_fp64
}

# Define mapping between tested function and numpy type
norm_fiber = {
    np.float32: nntile.nntile_core.tensor.norm_fiber_async_fp32,
    np.float64: nntile.nntile_core.tensor.norm_fiber_async_fp64
}

norm_fiber_inplace = {
    np.float32: nntile.nntile_core.tensor.norm_fiber_inplace_async_fp32,
    np.float64: nntile.nntile_core.tensor.norm_fiber_inplace_async_fp64
}


@pytest.mark.parametrize(
    'dtype, inplace, input_shape', [
        (np.float32, True, [3, 5, 20, 20]),
        (np.float64, True, [3, 5, 20, 20]),
        (np.float32, True, [7, 5, 21, 21]),
        (np.float64, True, [7, 5, 21, 21]),
        (np.float32, False, [3, 5, 20, 20]),
        (np.float64, False, [3, 5, 20, 20]),
        (np.float32, False, [7, 5, 21, 21]),
        (np.float64, False, [7, 5, 21, 21]),
    ]
)
def test_norm_fiber_async(dtype, inplace, input_shape):
    # Describe single-tile tensor, located at node 0
    mpi_distr = [0]
    next_tag = 0
    alpha = float(1.0)
    beta = float(0.0)
    next_tag = 0
    shape_A = np.array(input_shape)
    shape_B = shape_A[1:2]
    shape_C = shape_A[1:2]

    # data generation
    traits_A = nntile.tensor.TensorTraits(shape_A, shape_A)
    A = Tensor[dtype](traits_A, mpi_distr, next_tag)
    np_A = np.ones(shape_A, order='F').astype(dtype)
    A.from_array(np_A)
    next_tag = A.next_tag

    traits_B = nntile.tensor.TensorTraits(shape_B, shape_B)
    B = Tensor[dtype](traits_B, mpi_distr, next_tag)
    np_B = np.ones(shape_B, order='F').astype(dtype)
    B.from_array(np_B)
    next_tag = B.next_tag

    traits_C = nntile.tensor.TensorTraits(shape_C, shape_C)
    C = Tensor[dtype](traits_C, mpi_distr, next_tag)
    np_C = np.ones(shape_C, order='F').astype(dtype)
    C.from_array(np_C)
    next_tag = C.next_tag

    # acutal calculations
    if inplace:
        norm_fiber_inplace[dtype](alpha, A, beta, B, 1, 0, 0)
    else:
        norm_fiber[dtype](alpha, A, beta, C, B, 1, 0, 0)
    B.to_array(np_B)
    nntile.starpu.wait_for_all()

    # reference value
    ref = (input_shape[0] * input_shape[2] * input_shape[3]) ** 0.5

    A.unregister()
    B.unregister()
    C.unregister()

    assert np.allclose(np_B[0], ref)
