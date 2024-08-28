# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_randn.py
# Test for tensor::randn<T> Python wrapper
#
# @version 1.1.0

import numpy as np
import pytest

import nntile

config = nntile.starpu.Config(1, 0, 0)
nntile.starpu.init()

# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
          np.float64: nntile.tensor.Tensor_fp64}

# Define mapping between tested function and numpy type
randn = {np.float32: nntile.nntile_core.tensor.randn_fp32,
        np.float64: nntile.nntile_core.tensor.randn_fp64}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_randn(dtype):
    # Describe single-tile tensor, located at node 0
    shape = [100, 100, 100]
    ndim = len(shape)
    seed = 1
    mean = 1.0
    dev = 0.5
    mpi_distr = [0]
    next_tag = 0
    traits = nntile.tensor.TensorTraits(shape, shape)
    # Tensor objects
    A = Tensor[dtype](traits, mpi_distr, next_tag)
    # Set initial values of tensors
    randn[dtype](A, [0] * ndim, shape, seed, mean, dev)
    np_A = np.zeros(shape, dtype=dtype, order='F')
    A.to_array(np_A)
    nntile.starpu.wait_for_all()
    A.unregister()
    # Check average value and variation
    mean2 = np.mean(np_A)
    np_A -= mean2
    np_A *= np_A
    dev2 = np.mean(np_A) ** 0.5
    assert abs(1 - mean2 / mean) < 1e-3
    assert abs(1 - dev2 / dev) < 1e-3
