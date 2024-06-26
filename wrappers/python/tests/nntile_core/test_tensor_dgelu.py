# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_dgelu.py
# Test for tensor::dgelu<T> Python wrapper
#
# @version 1.0.0

# All necesary imports
import sys
import nntile
import math
import numpy as np
from scipy.special import erf
from numpy import sqrt
from numpy import tanh

# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
          np.float64: nntile.tensor.Tensor_fp64}
# Define mapping between tested function and numpy type
dgelu = {np.float32: nntile.nntile_core.tensor.dgelu_fp32,
        np.float64: nntile.nntile_core.tensor.dgelu_fp64}


def dgelu_numpy(x, approximate=True):
    """
    https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/
    approximated verison also should work, but need to fix tol in all close
    now test non-approx version
    """
    s = x / np.sqrt(2)
    erf_prime = lambda x: (2 / np.sqrt(np.pi)) * np.exp(-(x ** 2))  # noqa: E731
    if approximate:
        approx = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))
        dx = 0.5 + 0.5 * approx + ((0.5 * x * erf_prime(s)) / np.sqrt(2))
    else:
        dx = 0.5 + 0.5 * erf(s) + ((0.5 * x * erf_prime(s)) / np.sqrt(2))
    return dx


# Helper function returns bool value true if test passes
def helper(dtype, approximate=True):
    # Describe single-tile tensor, located at node 0
    shape = [2, 2]
    mpi_distr = [0]
    next_tag = 0
    traits = nntile.tensor.TensorTraits(shape, shape)
    # Tensor objects
    A = Tensor[dtype](traits, mpi_distr, next_tag)
    # Set initial values of tensors
    rand = np.random.randn(*shape)
    src_A = np.array(rand, dtype=dtype, order='F')
    dst_A = np.zeros_like(src_A)
    A.from_array(src_A)
    dgelu[dtype](A)
    A.to_array(dst_A)
    nntile.starpu.wait_for_all()
    A.unregister()
    # Get result in numpy
    src_A = dgelu_numpy(src_A, approximate=approximate)
    verbose = 'src_a {0} of {1}\ndst_A {2} of {1}\n'.format(src_A,dtype,dst_A)
    print(verbose)
    return np.allclose(src_A, dst_A)


# Test runner for different precisions
def test():
    for dtype in dtypes:
        for a in [False]:
            assert helper(dtype, approximate=a)


# Repeat tests
def test_repeat():
    for dtype in dtypes:
        for a in [False]:
            assert helper(dtype, approximate=a)

if __name__ == "__main__":
    test()
    test_repeat()
