# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_axpy.py
# Test for tensor::axpy<T> Python wrapper
#
# @version 1.0.0
# @author Aleksandr Katrutsa
# @author Aleksandr Mikhalev
# @date 2023-02-14

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
# Define mapping between tested function and numpy type
axpy = {np.float32: nntile.nntile_core.tensor.axpy_fp32,
        np.float64: nntile.nntile_core.tensor.axpy_fp64}


# Helper function returns bool value true if test passes
def helper(dtype):
    # Describe single-tile tensor, located at node 0
    shape = [2, 3, 4]
    mpi_distr = [0]
    next_tag = 0
    traits = nntile.tensor.TensorTraits(shape, shape)
    const_traits = nntile.tensor.TensorTraits([], [])
    # Tensor objects
    A = Tensor[dtype](traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    B = Tensor[dtype](traits, mpi_distr, next_tag)
    next_tag = B.next_tag
    alpha = Tensor[dtype](const_traits, mpi_distr, next_tag)
    # Set initial values of tensors
    rand_A = np.random.randn(*shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)
    rand_B = np.random.randn(*shape)
    np_B = np.array(rand_B, dtype=dtype, order='F')
    B.from_array(np_B)
    a = np.random.randn(1)
    alpha_np = np.array(a, dtype=dtype, order='F')
    alpha.from_array(alpha_np)
    axpy[dtype](alpha, A, B)
    np_C = np.zeros(shape, dtype=dtype, order='F')
    B.to_array(np_C)
    nntile.starpu.wait_for_all()
    A.unregister()
    B.unregister()
    alpha.unregister()
    # Compare results
    return np.allclose(np_C, rand_B + alpha_np * rand_A)

# Helper function returns bool value true if test passes
def helper2(dtype):
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
    rand_A = np.random.randn(*shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)
    rand_B = np.random.randn(*shape)
    np_B = np.array(rand_B, dtype=dtype, order='F')
    B.from_array(np_B)
    a = np.array(np.random.randn(1), dtype=dtype)
    axpy[dtype](a[0], A, B)
    np_C = np.zeros(shape, dtype=dtype, order='F')
    B.to_array(np_C)
    nntile.starpu.wait_for_all()
    A.unregister()
    B.unregister()
    # Compare results
    return np.allclose(np_C, rand_B + a * rand_A)

# Test runner for different precisions
def test():
    for dtype in dtypes:
        assert helper(dtype)
        assert helper2(dtype)

# Repeat tests
def test_repeat():
    for dtype in dtypes:
        assert helper(dtype)
        assert helper2(dtype)

if __name__ == "__main__":
    test()
    test_repeat()
