# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_loss_frob.py
# Test for nntile.loss.frob
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-02-08

# All necesary imports
import nntile
import numpy as np
# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()

Tensor_fp32 = nntile.tensor.Tensor_fp32
Frob_fp32 = nntile.loss.Frob_fp32

# Helper function returns bool value true if test passes
def helper_main_fp32():
    # Describe single-tile tensor, located at node 0
    A_shape = [4, 5, 6]
    ndim = len(A_shape)
    dtype = np.float32
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    mpi_distr = [0]
    next_tag = 0
    val_traits = nntile.tensor.TensorTraits([], [])
    # Tensor objects
    A = Tensor_fp32(A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    dA = Tensor_fp32(A_traits, mpi_distr, next_tag)
    next_tag = dA.next_tag
    B = Tensor_fp32(A_traits, mpi_distr, next_tag)
    next_tag = B.next_tag
    val = Tensor_fp32(val_traits, [0], next_tag)
    next_tag = val.next_tag
    # Set initial values of tensors
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)
    rand_B = np.random.randn(*A_shape)
    np_B = np.array(rand_B, dtype=dtype, order='F')
    B.from_array(np_B)
    # Define loss function object
    loss, next_tag = Frob_fp32.generate_block_cyclic(A, dA, B, val, next_tag)
    # Check loss and gradient
    loss.grad()
    np_dA = np.zeros_like(np_A)
    dA.to_array(np_dA)
    np_C = np_A - np_B
    if (np_C != np_dA).any():
        return False
    np_val = np.zeros([1], order='F', dtype=dtype)
    val.to_array(np_val)
    if not np.isclose(np_val, 0.5*np.linalg.norm(np_C)):
        return False
    # Clear data
    A.unregister()
    dA.unregister()
    B.unregister()
    val.unregister()
    return True

# Test runner for different precisions
def test():
    assert helper_main_fp32()

# Repeat tests
def test_repeat():
    assert helper_main_fp32()

if __name__ == "__main__":
    test()
    test_repeat()

