# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_copy_intersection.py
# Test for tensor::copy_intersection<T> Python wrapper
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
copy_intersection = {
        np.float32: nntile.nntile_core.tensor.copy_intersection_fp32,
        np.float64: nntile.nntile_core.tensor.copy_intersection_fp64}

# Helper function returns bool value true if test passes
def helper(dtype):
    # Describe single-tile tensor, located at node 0
    A_shape = [3, 4, 5]
    B_shape = [5, 4, 3]
    A_offset = [10, 10, 10]
    B_offset = [7, 10, 13]
    ndim = len(A_shape)
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    B_traits = nntile.tensor.TensorTraits(B_shape, B_shape)
    A_distr = [0]
    B_distr = [0]
    # Tensor objects
    next_tag = 0
    A = Tensor[dtype](A_traits, A_distr, next_tag)
    next_tag = A.next_tag
    B = Tensor[dtype](B_traits, B_distr, next_tag)
    # Set initial values of tensors
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)
    rand_B = np.random.randn(*B_shape)
    np_B = np.array(rand_B, dtype=dtype, order='F')
    B.from_array(np_B)
    # Check result
    copy_intersection[dtype](A, A_offset, B, B_offset)
    B.to_array(np_B)
    nntile.starpu.wait_for_all()
    A.unregister()
    B.unregister()
    return (np_A[:2, :, 3:] == np_B[3:, :, :2]).all()

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

