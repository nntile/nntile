# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/layer/test_linear.py
# Test for nntile.layer.linear
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-02-12

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
# Get multiprecision activation layer
Linear = nntile.layer.Linear

# Helper function returns bool value true if test passes
def helper(dtype: np.dtype):
    # Describe single-tile tensor, located at node 0
    A_shape = [4, 5, 6]
    ndim = len(A_shape)
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    mpi_distr = [0]
    next_tag = 0
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    dA = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = dA.next_tag
    # Set initial values of tensors
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A_moments = nntile.tensor.TensorMoments(A, dA, True)
    # Define linear layer
    layer, next_tag = Linear.generate_simple_mpiroot(A_moments, 'L',
            nntile.tensor.notrans, 2, [7, 8], [7, 8], next_tag)
    rand_W = np.random.randn(*layer.w.value.shape)
    np_W = np.array(rand_W, dtype=dtype, order='F')
    layer.w.value.from_array(np_W)
    # Check result
    A.from_array(np_A)
    layer.forward_async()
    np_Y = np.tensordot(np_A, np_W, 2)
    np_Y2 = np.zeros_like(np_Y, order='F')
    layer.y.value.to_array(np_Y2)
    if np.linalg.norm(np_Y-np_Y2)/np.linalg.norm(np_Y) > 1e-5:
        return False
    layer.y.grad.from_array(np_Y)
    layer.backward_async()
    np_Z = np.einsum("ijk,ilm->jklm", np_A, np_Y2)
    np_Z2 = np.zeros_like(np_Z, order='F')
    layer.w.grad.to_array(np_Z2)
    if np.linalg.norm(np_Z-np_Z2)/np.linalg.norm(np_Z) > 1e-5:
        return False
    np_Z3 = np.einsum("ijk,lmjk->ilm", np_Y2, np_W)
    np_Z4 = np.zeros_like(np_Z3, order='F')
    layer.x.grad.to_array(np_Z4)
    if np.linalg.norm(np_Z3-np_Z4)/np.linalg.norm(np_Z3) > 1e-5:
        return False
    A.unregister()
    dA.unregister()
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

