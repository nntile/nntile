# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_add_slice.py
# Test for tensor::add_slice<T> Python wrapper
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
# Define mapping between tested function and numpy type
add_slice = {np.float32: nntile.nntile_core.tensor.add_slice_fp32, \
        np.float64: nntile.nntile_core.tensor.add_slice_fp64}

# Helper function returns bool value true if test passes
def helper_axis(dtype):
    # Describe single-tile tensor, located at node 0
    A_shape = [2, 3, 4]
    B_shape = []
    ndim = len(A_shape)
    for i in range(ndim):
        B_shape.append(A_shape[:i]+A_shape[i+1:])
    mpi_distr = [0]
    next_tag = 0
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    B_traits = []
    for i in range(ndim):
        B_traits.append(nntile.tensor.TensorTraits(B_shape[i], B_shape[i]))
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    B = []
    for i in range(ndim):
        B.append(Tensor[dtype](B_traits[i], mpi_distr, next_tag))
        next_tag = B[-1].next_tag
    # Set initial values of tensors
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    np_A2 = np.zeros_like(np_A)
    np_B = []
    for i in range(ndim):
        rand_B = np.random.randn(*B_shape[i])
        np_B.append(np.array(rand_B, dtype=dtype, order='F'))
        B[i].from_array(np_B[-1])
    # Check result along each axis
    alpha = -0.5
    beta = 0.5
    for i in range(ndim):
        A.from_array(np_A)
        add_slice[dtype](alpha, B[i], beta, A, i)
        A.to_array(np_A2)
        nntile.starpu.wait_for_all()
        B[i].unregister()
        np_C = alpha * np.expand_dims(np_B[i], axis=i)
        np_C = np.repeat(np_C, A_shape[i], axis=i)
        np_C += beta * np_A
        nntile.starpu.wait_for_all()
        if not np.allclose(np_C, np_A2):
            return False
    A.unregister()
    return True

# Test runner for different precisions
def test():
    for dtype in dtypes:
        assert helper_axis(dtype)

# Repeat tests
def test_repeat():
    for dtype in dtypes:
        assert helper_axis(dtype)

if __name__ == "__main__":
    test()
    test_repeat()

