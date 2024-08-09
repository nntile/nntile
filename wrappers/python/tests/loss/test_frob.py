# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/loss/test_frob.py
# Test for nntile.loss.frob
#
# @version 1.1.0

import numpy as np
import pytest
from numpy.testing import assert_equal

import nntile
from nntile.loss import Frob

config = nntile.starpu.Config(1, 0, 0)
nntile.starpu.init()

# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
        np.float64: nntile.tensor.Tensor_fp64}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_frob(dtype: np.dtype):
    """Helper function returns bool value true if test passes."""
    rng = np.random.default_rng(42)

    # Describe single-tile tensor, located at node 0
    A_shape = [4, 5, 6]
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    mpi_distr = [0]
    next_tag = 0
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Set initial values of tensors
    rand_A = rng.standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)
    # Define loss function object
    loss, next_tag = Frob.generate_simple(A_moments, next_tag)
    # Check loss and gradient
    rand_B = rng.standard_normal(A_shape)
    np_B = np.array(rand_B, dtype=dtype, order='F')
    loss.y.from_array(np_B)
    loss.calc_async()
    np_A_grad = np.zeros_like(np_A)
    A_moments.grad.to_array(np_A_grad)
    np_C = np_A - np_B
    assert_equal(np_C, np_A_grad)
    np_val = np.zeros([1], order='F', dtype=dtype)
    loss.val.to_array(np_val)
    assert np.isclose(np_val, 0.5 * np.linalg.norm(np_C) ** 2)
    # Clear data
    A_moments.unregister()
