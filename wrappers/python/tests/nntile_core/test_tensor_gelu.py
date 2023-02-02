# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_gelu.py
# Test for tensor::gelu<T> Python wrapper
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @author Konstantin Sozykin
# @date 2023-02-02

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
gelu = {np.float32: nntile.tensor.gelu_fp32,
        np.float64: nntile.tensor.gelu_fp64}


def gelu_numpy(z, approximate=True):
    """
    https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/
    approximated verison also should work, but need to fix tol in all close
    now test non-approx version
    """
    if approximate:
        return 0.5 * z * (1 + tanh(sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)))
    # math,erf is not vectorized, sp.special.erf
    return 0.5 * z * (1 + erf(z / sqrt(2)))


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
    gelu[dtype](A)
    A.to_array(dst_A)
    nntile.starpu.wait_for_all()
    A.unregister()
    # Get result in numpy
    src_A = gelu_numpy(src_A, approximate=approximate)
    print(f'src_a {src_A} of {dtype}\ndst_A {dst_A} of {dtype}\n')
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
