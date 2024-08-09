# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_copy_intersection.py
# Test for tensor::copy_intersection<T> Python wrapper
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
copy_intersection = {
    np.float32: nntile.nntile_core.tensor.copy_intersection_fp32,
    np.float64: nntile.nntile_core.tensor.copy_intersection_fp64}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_clear(dtype):
    # Describe single-tile tensor, located at node 0
    A_shape = [3, 4, 5]
    B_shape = [5, 4, 3]
    A_offset = [10, 10, 10]
    B_offset = [7, 10, 13]
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    B_traits = nntile.tensor.TensorTraits(B_shape, B_shape)
    A_distr = [0]
    B_distr = [0]
    # Tensor objects
    next_tag = 0
    A = Tensor[dtype](A_traits, A_distr, next_tag)
    next_tag = A.next_tag
    B = Tensor[dtype](B_traits, B_distr, next_tag)
    # Set initial values of tensors

    rng = np.random.default_rng(42)
    rand_A = rng.standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)
    rand_B = rng.standard_normal(B_shape)
    np_B = np.array(rand_B, dtype=dtype, order='F')
    B.from_array(np_B)
    # Check result
    copy_intersection[dtype](A, A_offset, B, B_offset)
    B.to_array(np_B)
    nntile.starpu.wait_for_all()
    A.unregister()
    B.unregister()
    assert_equal(np_A[:2, :, 3:], np_B[3:, :, :2])
