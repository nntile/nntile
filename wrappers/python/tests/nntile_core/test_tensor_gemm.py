# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_gemm.py
# Test for tensor::gemm<T> Python wrapper
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
gemm = {np.float32: nntile.nntile_core.tensor.gemm_fp32,
        np.float64: nntile.nntile_core.tensor.gemm_fp64}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_gemm(dtype):
    # Describe single-tile tensor, located at node 0
    matrix_shape = [2, 2]
    batch = 3
    mpi_distr = [0]
    next_tag = 0
    shape = [*matrix_shape, batch]
    traits = nntile.tensor.TensorTraits(shape, shape)
    # Tensor objects
    A = Tensor[dtype](traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    B = Tensor[dtype](traits, mpi_distr, next_tag)
    next_tag = B.next_tag
    C = Tensor[dtype](traits, mpi_distr, next_tag)
    # Set initial values of tensors
    rng = np.random.default_rng(42)
    src_A = rng.standard_normal(shape).astype(dtype, 'F')
    src_B = rng.standard_normal(shape).astype(dtype, 'F')
    src_C = rng.standard_normal(shape).astype(dtype, 'F')
    dst_C = np.zeros_like(src_C)
    A.from_array(src_A)
    B.from_array(src_B)
    C.from_array(src_C)
    # Get results by means of nntile and convert to numpy
    alpha = 1
    beta = -1
    gemm[dtype](alpha, nntile.notrans, A, nntile.trans, B, beta, C, 1, 1, 0)
    C.to_array(dst_C)
    nntile.starpu.wait_for_all()
    A.unregister()
    B.unregister()
    C.unregister()
    # Check results
    for i in range(batch):
        # Get result in numpy
        src_C[:, :, i] = (beta * src_C[:, :, i] + alpha *
                          (src_A[:, :, i] @ (src_B[:, :, i].T)))
        # Check if results are almost equal
        rel_error = (np.linalg.norm(dst_C[:, :, i] - src_C[:, :, i]) /
                     np.linalg.norm(src_C[:, :, i]))
        assert rel_error <= 1e-4
