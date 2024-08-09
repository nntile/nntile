# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_addcdiv.py
# Test for tensor::addcdiv<T> Python wrapper
#
# @version 1.1.0

import numpy as np
import pytest
from numpy.testing import assert_allclose

import nntile

# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
          np.float64: nntile.tensor.Tensor_fp64}

# Define mapping between tested function and numpy type
addcdiv = {np.float32: nntile.nntile_core.tensor.addcdiv_fp32,
           np.float64: nntile.nntile_core.tensor.addcdiv_fp64}

config = nntile.starpu.Config(1, 0, 0)
nntile.starpu.init()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_addcdiv(dtype):
    # Describe single-tile tensor, located at node 0
    shape = [2, 3, 4]
    mpi_distr = [0]
    next_tag = 0
    traits = nntile.tensor.TensorTraits(shape, shape)
    # Tensor objects
    A = Tensor[dtype](traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    B = Tensor[dtype](traits, mpi_distr, next_tag)
    next_tag = B.next_tag
    C = Tensor[dtype](traits, mpi_distr, next_tag)
    next_tag = C.next_tag

    # Set initial values of tensors
    rng = np.random.default_rng(42)

    rand_A = rng.standard_normal(shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)

    rand_B = rng.standard_normal(shape)
    np_B = np.array(rand_B, dtype=dtype, order='F')
    B.from_array(np_B)

    rand_C = rng.standard_normal(shape)
    np_C = np.array(rand_C, dtype=dtype, order='F')
    C.from_array(np_C)

    a = dtype(rng.standard_normal())
    eps = 1e-4
    addcdiv[dtype](a, eps, A, B, C)
    np_D = np.zeros(shape, dtype=dtype, order='F')
    C.to_array(np_D)
    nntile.starpu.wait_for_all()
    A.unregister()
    B.unregister()
    C.unregister()

    if dtype == np.float32:
        atol = 1e-5
    elif dtype == np.float64:
        atol = 1e-10
    assert_allclose(np_D, rand_C + a * rand_A / (rand_B + eps), atol=atol)
