# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_add_scalar.py
# Test for tensor::add_scalar<T> Python wrapper
#
# @version 1.1.0

import numpy as np
import pytest
from numpy.testing import assert_equal

import nntile

config = nntile.starpu.Config(1, 0, 0)
nntile.starpu.init()

# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
        np.float64: nntile.tensor.Tensor_fp64}
# Define mapping between tested function and numpy type
add_scalar = {np.float32: nntile.nntile_core.tensor.add_scalar_fp32,
        np.float64: nntile.nntile_core.tensor.add_scalar_fp64}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_add_scalar(dtype):
    # Describe single-tile tensor, located at node 0
    shape = [2, 3, 4]
    mpi_distr = [0]
    next_tag = 0
    alpha = 10.
    beta = -5.5
    traits = nntile.tensor.TensorTraits(shape, shape)
    # Tensor objects
    A = Tensor[dtype](traits, mpi_distr, next_tag)
    # Set initial values of tensors
    rand_A = np.random.default_rng(42).standard_normal(shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)
    add_scalar[dtype](alpha, beta, A)
    np_C = np.zeros(shape, dtype=dtype, order='F')
    A.to_array(np_C)
    nntile.starpu.wait_for_all()
    A.unregister()
    # Compare results
    assert_equal(alpha + beta * np_A, np_C)
