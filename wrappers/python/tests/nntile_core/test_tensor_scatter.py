# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_scatter.py
# Test for tensor::scatter<T> Python wrapper
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
scatter = {np.float32: nntile.nntile_core.tensor.scatter_fp32,
           np.float64: nntile.nntile_core.tensor.scatter_fp64}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_scatter(dtype):
    # Describe single-tile tensor, located at node 0
    A_shape = [2, 3, 4]
    B_basetile = [2, 2, 2]
    ndim = len(A_shape)
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    B_traits = nntile.tensor.TensorTraits(A_shape, B_basetile)
    A_distr = [0]
    B_distr = [0] * B_traits.grid.nelems
    # Tensor objects
    next_tag = 0
    A = Tensor[dtype](A_traits, A_distr, next_tag)
    next_tag = A.next_tag
    B = Tensor[dtype](B_traits, B_distr, next_tag)
    # Set initial values of tensors
    rand_A = np.random.default_rng(42).standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)
    # Check result
    scatter[dtype](A, B)
    A.unregister()
    for i in range(B_traits.grid.nelems):
        B_tile = B.get_tile(i)
        np_B_tile = np.zeros(B_tile.shape, dtype=dtype, order='F')
        B_tile.to_array(np_B_tile)
        B_tile_index = B.grid.linear_to_index(i)
        np_C = np_A.reshape([1] + A_shape)
        for j in range(ndim):
            ind0 = B_tile_index[j] * B_basetile[j]
            ind1 = ind0 + B_basetile[j]
            if ind1 > A_shape[j]:
                ind1 = A_shape[j]
            np_C = np_C[:, ind0:ind1, ...]
            new_shape = [np_C.shape[0] * np_C.shape[1]]
            new_shape.extend(np_C.shape[2:])
            np_C = np_C.reshape(new_shape)
        np_C = np_C.reshape(B_tile.shape)
        assert_equal(np_B_tile, np_C)
    B.unregister()
