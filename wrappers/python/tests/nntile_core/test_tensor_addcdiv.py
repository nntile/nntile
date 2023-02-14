# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_addcdiv.py
# Test for tensor::addcdiv<T> Python wrapper
#
# @version 1.0.0
# @author Aleksandr Katrutsa
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

addcdiv = {np.float32: nntile.tensor.addcdiv_fp32,
        np.float64: nntile.tensor.addcdiv_fp64}


# Helper function returns bool value true if test passes
def helper(dtype):
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
    rand_A = np.random.randn(*shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)
    
    rand_B = np.random.randn(*shape)
    np_B = np.array(rand_B, dtype=dtype, order='F')
    B.from_array(np_B)

    rand_C = np.random.randn(*shape)
    np_C = np.array(rand_C, dtype=dtype, order='F')
    C.from_array(np_C)

    a = np.array(np.random.randn(1), dtype=dtype)
    eps = 1e-4
    addcdiv[dtype](a[0], eps, A, B, C)
    np_D = np.zeros(shape, dtype=dtype, order='F')
    C.to_array(np_D)
    nntile.starpu.wait_for_all()
    A.unregister()
    B.unregister()
    C.unregister()
    # Compare results
    return np.allclose(np_D, rand_C + a * rand_A / (rand_B + eps))

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
