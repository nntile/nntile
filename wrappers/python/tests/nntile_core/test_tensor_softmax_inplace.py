# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_softmax_inplace.py
# Test for tensor::softmax_inplace<T> Python wrapper
#
# @version 1.1.0

import numpy as np
import pytest

import nntile

config = nntile.starpu.Config(1, 0, 0)
nntile.starpu.init()

# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
          np.float64: nntile.tensor.Tensor_fp64}

# Define mapping between tested function and numpy type
softmax_inplace = {np.float32: nntile.nntile_core.tensor.softmax_inplace_fp32,
                   np.float64: nntile.nntile_core.tensor.softmax_inplace_fp64}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_softmax_inplace(dtype):
    # Describe single-tile tensor, located at node 0
    A_shape = [2, 3, 4]
    B_shape = []
    ndim = len(A_shape)
    for i in range(ndim):
        B_shape.append([2] + A_shape[:i] + A_shape[i + 1:])
    mpi_distr = [0]
    next_tag = 0
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    B_traits = []
    for i in range(ndim):
        B_traits.append(nntile.tensor.TensorTraits(B_shape[i], B_shape[i]))
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    B = []
    for i in range(ndim):
        B.append(Tensor[dtype](B_traits[i], mpi_distr, next_tag))
        next_tag = B[-1].next_tag
    # Set initial values of tensors
    rng = np.random.default_rng(42)
    rand_A = rng.standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    np_A2 = np.zeros_like(np_A)
    np_B = []
    for i in range(ndim):
        rand_B = rng.standard_normal(B_shape[i])
        np_B.append(np.array(rand_B, dtype=dtype, order='F'))
        B[i].from_array(np_B[-1])
    # Check result along each axis
    for i in range(ndim):
        A.from_array(np_A)
        softmax_inplace[dtype](B[i], 1.0, A, i)
        B[i].unregister()
        A.to_array(np_A2)
        np_B_max = np.expand_dims(np_B[i][0, ...], axis=i)
        np_B_max = np.repeat(np_B_max, A_shape[i], axis=i)
        np_B_sumexp = np.expand_dims(np_B[i][1, ...], axis=i)
        np_B_sumexp = np.repeat(np_B_sumexp, A_shape[i], axis=i)
        np_C = (np.e ** (np_A - np_B_max)) / np_B_sumexp
        nntile.starpu.wait_for_all()
        assert np.allclose(np_C, np_A2)
    A.unregister()
