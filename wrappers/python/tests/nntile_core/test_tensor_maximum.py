# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_maximum.py
# Test for tensor::maximum<T> Python wrapper
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
maximum_func = {
    np.float32: nntile.nntile_core.tensor.maximum_fp32,
    np.float64: nntile.nntile_core.tensor.maximum_fp64}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_maximum(context, dtype):
    # Describe single-tile tensor, located at node 0
    shape = [2, 3, 4]
    traits = nntile.tensor.TensorTraits(shape, shape)
    # Tensor objects
    A = Tensor[dtype](traits)
    B = Tensor[dtype](traits)
    # Set initial values of tensors
    rng = np.random.default_rng(42)
    rand_A = rng.standard_normal(shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)
    rand_B = rng.standard_normal(shape)
    np_B = np.array(rand_B, dtype=dtype, order='F')
    B.from_array(np_B)
    maximum_func[dtype](A, B)
    np_C = np.zeros(shape, dtype=dtype, order='F')
    B.to_array(np_C)
    nntile.starpu.wait_for_all()
    A.unregister()
    B.unregister()
    # Compare results
    assert_equal(np_C, np.maximum(np_A, np_B))
