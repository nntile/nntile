# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_layer_norm.py
# Test for nntile.layer.norm
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-02-09

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
# Get multiprecision normalization layer
Norm = nntile.layer.Norm

# Helper function returns bool value true if test passes
def helper(dtype: np.dtype):
    # Describe single-tile tensor, located at node 0
    A_shape = [4, 5, 6]
    new_A_shape = [4, 8, 6]
    ndim = len(A_shape)
    eps = 1e-6
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    new_A_traits = nntile.tensor.TensorTraits(new_A_shape, new_A_shape)
    mpi_distr = [0]
    next_tag = 0
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    dA = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = dA.next_tag
    new_A = Tensor[dtype](new_A_traits, mpi_distr, next_tag)
    next_tag = new_A.next_tag
    new_dA = Tensor[dtype](new_A_traits, mpi_distr, next_tag)
    next_tag = new_dA.next_tag
    # Set initial values of tensors
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    rand_new_A = np.random.randn(*new_A_shape)
    np_new_A = np.array(rand_new_A, dtype=dtype, order='F')
    np_B = np.zeros_like(np_A)
    np_new_B = np.zeros_like(np_new_A)
    # Check result along each axis
    for i in range(ndim):
        # A is invalidated after each forward_async
        A.from_array(np_A)
        # Set up normalization layer
        layer, next_tag = Norm.generate_block_cyclic(A, dA, i, eps, \
                next_tag)
        # Do forward pass and wait until it is finished
        layer.forward_async()
        nntile.starpu.wait_for_all()
        # Dump output
        layer.y.to_array(np_B)
        # Check output correctness
        np_A_mean = np_A.mean(axis=i, keepdims=True)
        np_C = np_A - np_A_mean
        np_A_dev = (np_A.var(axis=i, keepdims=True)+eps) ** 0.5
        np_C /= np_A_dev
        if not np.allclose(np_C, np_B, rtol=1e-4, atol=1e-5):
            return False
        # new_A is invalidated after each forward_async
        new_A.from_array(np_new_A)
        # Rebatch layer
        new_layer, next_tag = layer.rebatch(new_A, new_dA, 1, next_tag)
        # Do forward pass and wait until it is finished
        new_layer.forward_async()
        nntile.starpu.wait_for_all()
        # Dump output
        new_layer.y.to_array(np_new_B)
        # Check output correctness
        np_new_A_mean = np_new_A.mean(axis=i, keepdims=True)
        np_new_C = np_new_A - np_new_A_mean
        np_new_A_dev = (np_new_A.var(axis=i, keepdims=True)+eps) ** 0.5
        np_new_C /= np_new_A_dev
        if not np.allclose(np_new_C, np_new_B, rtol=1e-4, atol=1e-5):
            return False
    A.unregister()
    dA.unregister()
    new_A.unregister()
    new_dA.unregister()
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

