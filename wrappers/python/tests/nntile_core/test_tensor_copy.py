# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_copy.py
# Test for tensor::copy<T> Python wrapper
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
copy = {np.float32: nntile.nntile_core.tensor.copy_fp32,
        np.float64: nntile.nntile_core.tensor.copy_fp64}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_copy(dtype):
    # Describe single-tile tensor, located at node 0
    A_shape = [2, 3, 4]
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    A_distr = [0]
    # Tensor objects
    next_tag = 0
    A = Tensor[dtype](A_traits, A_distr, next_tag)
    next_tag = A.next_tag
    B = Tensor[dtype](A_traits, A_distr, next_tag)
    # Set initial values of tensors
    rand_A = np.random.default_rng(42).standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)
    # Check result
    copy[dtype](A, B)
    np_B = np.zeros_like(np_A)
    B.to_array(np_B)
    nntile.starpu.wait_for_all()
    A.unregister()
    B.unregister()
    assert_equal(np_A, np_B)
