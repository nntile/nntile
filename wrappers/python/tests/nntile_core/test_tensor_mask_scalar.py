# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_mask_scalar.py
# Test for tensor::mask_scalar<T> Python wrapper
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
          np.float64: nntile.tensor.Tensor_fp64,
          bool: nntile.tensor.Tensor_bool}

# Define mapping between tested function and numpy type
mask_scalar_func = {np.float32: nntile.nntile_core.tensor.mask_scalar_fp32,
                    np.float64: nntile.nntile_core.tensor.mask_scalar_fp64}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_mask_scalar(dtype):
    # Describe single-tile tensor, located at node 0
    shape = [3, 3, 10]
    mpi_distr = [0]
    next_tag = 0
    traits = nntile.tensor.TensorTraits(shape, shape)
    # Tensor objects
    A = Tensor[dtype](traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    mask_traits = nntile.tensor.TensorTraits(shape[:2], shape[:2])
    mask = Tensor[bool](mask_traits, mpi_distr, next_tag)
    # Set initial values of tensors
    rand_A = np.random.default_rng(42).standard_normal(shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)

    causal_mask = np.tril(np.ones((3, 3), dtype=bool))
    mask_value = -1000.
    np_res = np.where(causal_mask[:, :, np.newaxis], np_A, mask_value)

    mask.from_array(np.array(causal_mask, dtype=bool, order="F"))
    mask_scalar_func[dtype](mask, dtype(mask_value), A, 1)
    A.to_array(np_A)
    nntile.starpu.wait_for_all()
    A.unregister()
    mask.unregister()
    # Compare results
    assert_equal(np_res, np_A)
