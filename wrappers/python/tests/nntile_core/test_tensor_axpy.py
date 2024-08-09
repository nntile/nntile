# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_axpy.py
# Test for tensor::axpy<T> Python wrapper
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
axpy = {np.float32: nntile.nntile_core.tensor.axpy_fp32,
        np.float64: nntile.nntile_core.tensor.axpy_fp64}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('scalar_type', ['nntile', 'numpy'])
def test_axpy(dtype, scalar_type: str):
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

    # Set initial values of tensors
    rng = np.random.default_rng(42)
    rand_A = rng.standard_normal(shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)
    rand_B = rng.standard_normal(shape)
    np_B = np.array(rand_B, dtype=dtype, order='F')
    B.from_array(np_B)

    match scalar_type:
        case 'nntile':
            const_traits = nntile.tensor.TensorTraits([], [])
            alpha_np = rng.standard_normal(1).astype(dtype, 'F')
            alpha = Tensor[dtype](const_traits, mpi_distr, next_tag)
            alpha.from_array(alpha_np)
        case 'numpy':
            alpha_np = dtype(rng.standard_normal())
            alpha = alpha_np

    axpy[dtype](alpha, A, B)
    np_C = np.zeros(shape, dtype=dtype, order='F')
    B.to_array(np_C)
    nntile.starpu.wait_for_all()
    A.unregister()
    B.unregister()
    if scalar_type == 'nntile':
        alpha.unregister()
    # Compare results
    assert np.allclose(np_C, rand_B + alpha_np * rand_A)
