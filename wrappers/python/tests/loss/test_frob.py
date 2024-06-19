# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/loss/test_frob.py
# Test for nntile.loss.frob
#
# @version 1.0.0

# All necesary imports
import nntile
import numpy as np
# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
        np.float64: nntile.tensor.Tensor_fp64}
# Get multiprecision loss function
Frob = nntile.loss.Frob

# Helper function returns bool value true if test passes
def helper(dtype: np.dtype):
    # Describe single-tile tensor, located at node 0
    A_shape = [4, 5, 6]
    ndim = len(A_shape)
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    mpi_distr = [0]
    next_tag = 0
    val_traits = nntile.tensor.TensorTraits([], [])
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Set initial values of tensors
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)
    # Define loss function object
    loss, next_tag = Frob.generate_simple(A_moments, next_tag)
    # Check loss and gradient
    rand_B = np.random.randn(*A_shape)
    np_B = np.array(rand_B, dtype=dtype, order='F')
    loss.y.from_array(np_B)
    loss.calc_async()
    np_A_grad = np.zeros_like(np_A)
    A_moments.grad.to_array(np_A_grad)
    np_C = np_A - np_B
    if (np_C != np_A_grad).any():
        return False
    np_val = np.zeros([1], order='F', dtype=dtype)
    loss.val.to_array(np_val)
    if not np.isclose(np_val, 0.5*np.linalg.norm(np_C)**2):
        return False
    # Clear data
    A_moments.unregister()
    return True

# Test runner for different precisions
def test():
    for dtype in dtypes:
        assert helper(dtype)

# Repeat tests
def test_repeat():
    for dtype in dtypes:
        assert helper(dtype)

if __name__ == "__main__":
    test()
    test_repeat()

