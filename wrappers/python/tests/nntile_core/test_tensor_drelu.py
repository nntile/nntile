# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_drelu.py
# Test for tensor::drelu<T> Python wrapper
#
# @version 1.1.0

import numpy as np
import pytest
from numpy.testing import assert_equal

import nntile

# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
          np.float64: nntile.tensor.Tensor_fp64}

# Define mapping between tested function and numpy type
drelu = {np.float32: nntile.nntile_core.tensor.drelu_fp32,
         np.float64: nntile.nntile_core.tensor.drelu_fp64}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_drelu(context, dtype):
    # Describe single-tile tensor, located at node 0
    shape = [2, 2]
    traits = nntile.tensor.TensorTraits(shape, shape)
    # Tensor objects
    A = Tensor[dtype](traits)
    # Set initial values of tensors
    rand = np.random.default_rng(42).standard_normal(shape)
    src_A = np.array(rand, dtype=dtype, order='F')
    dst_A = -np.ones_like(src_A)
    A.from_array(src_A)
    drelu[dtype](A)
    A.to_array(dst_A)
    nntile.starpu.wait_for_all()
    A.unregister()
    # Get result in numpy
    src_A[src_A < 0] = 0
    src_A[src_A > 0] = 1
    assert_equal(src_A, dst_A)
