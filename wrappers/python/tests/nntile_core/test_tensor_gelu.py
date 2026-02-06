# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_gelu.py
# Test for tensor::gelu<T> Python wrapper
#
# @version 1.1.0

import math

import numpy as np
import pytest
from numpy import sqrt, tanh
from scipy.special import erf

import nntile

# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
          np.float64: nntile.tensor.Tensor_fp64}
# Define mapping between tested function and numpy type
gelu = {np.float32: nntile.nntile_core.tensor.gelu_fp32,
        np.float64: nntile.nntile_core.tensor.gelu_fp64}


def gelu_numpy(z, approximate=True):
    """
    https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/
    approximated verison also should work, but need to fix tol in all close
    now test non-approx version
    """
    if approximate:
        zs = 0.5 * z * (1 + tanh(sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)))
        return zs
    # math,erf is not vectorized, sp.special.erf
    return 0.5 * z * (1 + erf(z / sqrt(2)))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_gelu(context, dtype, approximate=False):
    # Describe single-tile tensor, located at node 0
    shape = [2, 2]
    traits = nntile.tensor.TensorTraits(shape, shape)
    # Tensor objects
    A = Tensor[dtype](traits, [0])
    B = Tensor[dtype](traits, [0])
    # Generate input data
    rand = np.random.default_rng(42).standard_normal(shape)
    src_A = np.array(rand, dtype=dtype, order='F')
    dst_B = np.zeros_like(src_A)
    A.from_array(src_A)
    B.from_array(dst_B)
    # Apply gelu
    gelu[dtype](A, B)
    B.to_array(dst_B)
    nntile.starpu.wait_for_all()
    # Cleanup
    A.unregister()
    B.unregister()
    # Compute reference result
    src_A_ref = gelu_numpy(src_A, approximate=approximate)
    # Compare
    assert np.allclose(src_A_ref, dst_B)
