# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_relu.py
# Test for tensor::relu<T> Python wrapper
#
# @version 1.1.0

import numpy as np
import pytest

import nntile

nntile.nntile_init(ncpus=1, ncuda=0, cublas=0, ooc=0, logger=0, verbose=0)

# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
          np.float64: nntile.tensor.Tensor_fp64}

# Define mapping between tested function and numpy type
relu = {np.float32: nntile.nntile_core.tensor.relu_fp32,
        np.float64: nntile.nntile_core.tensor.relu_fp64}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_relu(dtype):
    # Describe single-tile tensor, located at node 0
    shape = [2, 2]
    traits = nntile.tensor.TensorTraits(shape, shape)
    # Tensor objects
    A = Tensor[dtype](traits)
    # Set initial values of tensors
    rand = np.random.default_rng(42).standard_normal(shape)
    src_A = np.array(rand, dtype=dtype, order='F')
    dst_A = np.zeros_like(src_A)
    A.from_array(src_A)
    relu[dtype](A)
    A.to_array(dst_A)
    nntile.starpu.wait_for_all()
    A.unregister()
    src_A = np.maximum(0, src_A)
    assert np.allclose(src_A, dst_A)
