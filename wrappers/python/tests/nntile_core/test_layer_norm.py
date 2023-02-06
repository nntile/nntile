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
# @date 2023-02-06

# All necesary imports
import nntile
import numpy as np
# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()

Tensor_fp32 = nntile.tensor.Tensor_fp32
Norm_fp32 = nntile.layer.Norm_fp32

# Helper function returns bool value true if test passes
def helper_fp32():
    # Describe single-tile tensor, located at node 0
    A_shape = [2, 3, 4]
    ndim = len(A_shape)
    dtype = np.float32
    eps = 1e-6
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    mpi_distr = [0]
    next_tag = 0
    # Tensor objects
    A = Tensor_fp32(A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    dA = Tensor_fp32(A_traits, mpi_distr, next_tag)
    next_tag = dA.next_tag
    # Set initial values of tensors
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    np_B = np.zeros_like(np_A)
    # Check result along each axis
    for i in range(ndim):
        # A is invalidate after each forward_async
        A.from_array(np_A)
        # Set up normalization layer
        layer, next_tag = Norm_fp32.generate_block_cyclic(A, dA, i, eps,
                next_tag)
        layer.forward_async()
        nntile.starpu.wait_for_all()
        layer.y.to_array(np_B)
        np_A_mean = np_A.mean(axis=i, keepdims=True)
        np_C = np_A - np_A_mean
        np_A_dev = (np_A.var(axis=i, keepdims=True)+eps)**0.5
        np_C /= np_A_dev
        if not np.allclose(np_C, np_B, rtol=4e-2, atol=1e-5):
            import pdb
            pdb.set_trace()
            return False
    A.unregister()
    dA.unregister()
    return True

# Test runner for different precisions
def test():
    assert helper_fp32()

# Repeat tests
def test_repeat():
    assert helper_fp32()

if __name__ == "__main__":
    test()
    test_repeat()

