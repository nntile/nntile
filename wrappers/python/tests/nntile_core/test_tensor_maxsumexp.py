# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_maxsumexp.py
# Test for tensor::maxsumexp<T> Python wrapper
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
maxsumexp = {np.float32: nntile.nntile_core.tensor.maxsumexp_fp32,
             np.float64: nntile.nntile_core.tensor.maxsumexp_fp64}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_maxsumexp(dtype):
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
    rand_A = np.random.default_rng(42).standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)
    np_B = []
    for i in range(ndim):
        np_B.append(np.zeros(B_shape[i], dtype=dtype, order='F'))
        B[i].from_array(np_B[-1])
    # Check result along each axis
    for i in range(ndim):
        nntile.tensor.clear_async(B[i])
        maxsumexp[dtype](A, B[i], i, 0)
        B[i].to_array(np_B[i])
        nntile.starpu.wait_for_all()
        B[i].unregister()
        np_max = np.max(np_A, axis=i)
        assert np.allclose(np_B[i][0, ...], np_max)
        np_max_expanded = np.expand_dims(np_max, axis=i)
        np_sumexp = np_A - np.repeat(np_max_expanded, A_shape[i], axis=i)
        np_sumexp = np.e ** np_sumexp
        np_sumexp = np.sum(np_sumexp, axis=i)
        assert np.allclose(np_B[i][1, ...], np_sumexp)
    A.unregister()
