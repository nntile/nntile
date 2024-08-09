# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_normalize.py
# Test for tensor::normalize<T> Python wrapper
#
# @version 1.1.0

import numpy as np
import pytest
from numpy.testing import assert_allclose

import nntile

config = nntile.starpu.Config(1, 0, 0)
nntile.starpu.init()

# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
          np.float64: nntile.tensor.Tensor_fp64}

# Define mapping between tested function and numpy type
normalize = {np.float32: nntile.nntile_core.tensor.normalize_fp32,
             np.float64: nntile.nntile_core.tensor.normalize_fp64}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_normalize(dtype):
    # Describe single-tile tensor, located at node 0
    A_shape = [2, 3, 4]
    eps = 1e-6
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
    gb_traits = nntile.tensor.TensorTraits([2], [2])
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    B = []
    for i in range(ndim):
        B.append(Tensor[dtype](B_traits[i], mpi_distr, next_tag))
        next_tag = B[-1].next_tag
    gamma_beta = Tensor[dtype](gb_traits, mpi_distr, next_tag)
    # Set initial values of tensors
    rng = np.random.default_rng(42)
    rand_A = rng.standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    np_A2 = np.zeros_like(np_A)
    np_B = []
    for i in range(ndim):
        # Generate random sums
        rand_B_sum = rng.standard_normal(B_shape[i][1:])
        # Generate random norms, but mean square must be no less than square of
        # mean
        rand_B_norm = np.abs(rand_B_sum) / (A_shape[i] ** 0.5)
        rand_B_norm += np.abs(rng.standard_normal(B_shape[i][1:]))
        rand_B = [rand_B_sum, rand_B_norm]
        np_B.append(np.array(rand_B, dtype=dtype, order='F'))
        del rand_B, rand_B_sum, rand_B_norm
        B[i].from_array(np_B[-1])
    np_gb = np.array([2.0, 1.0], dtype=dtype, order='F')
    gamma_beta.from_array(np_gb)
    # Check result along each axis
    for i in range(ndim):
        A.from_array(np_A)
        normalize[dtype](gamma_beta, B[i], A, A_shape[i], eps, i)
        A.to_array(np_A2)
        nntile.starpu.wait_for_all()
        B[i].unregister()
        np_B_mean = np.expand_dims(np_B[i][0, ...], axis=i) / A_shape[i]
        np_B_sqrmean = np.expand_dims(np_B[i][1, ...], axis=i)**2 / A_shape[i]
        np_B_dev = (np_B_sqrmean - np_B_mean ** 2 + eps) ** 0.5
        np_B_mean = np.repeat(np_B_mean, A_shape[i], axis=i)
        np_B_dev = np.repeat(np_B_dev, A_shape[i], axis=i)
        np_C = (np_A - np_B_mean) / np_B_dev
        np_C = np_C * np_gb[0] + np_gb[1]
        nntile.starpu.wait_for_all()
        assert_allclose(np_C, np_A2, rtol=1e-3, atol=1e-5)
    A.unregister()
    gamma_beta.unregister()
