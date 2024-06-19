# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_nrm2.py
# Test for tensor::nrm2<T> Python wrapper
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
nrm2 = {np.float32: nntile.nntile_core.tensor.nrm2_fp32,
        np.float64: nntile.nntile_core.tensor.nrm2_fp64}

# Helper function returns bool value true if test passes
def helper(dtype):
    # Describe single-tile tensor, located at node 0
    shape = [2, 3, 4]
    ndim = len(shape)
    mpi_distr = [0]
    next_tag = 0
    A_traits = nntile.tensor.TensorTraits(shape, shape)
    B_traits = nntile.tensor.TensorTraits([], [])
    tmp_traits = nntile.tensor.TensorTraits([1]*ndim, [1]*ndim)
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    B = Tensor[dtype](B_traits, mpi_distr, next_tag)
    next_tag = B.next_tag
    tmp = Tensor[dtype](tmp_traits, mpi_distr, next_tag)
    # Set initial values of tensors
    rand_A = np.random.randn(*shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.from_array(np_A)
    nrm2[dtype](1.0, A, 0.0, B, tmp)
    np_B = np.zeros([1], dtype=dtype, order='F')
    B.to_array(np_B)
    nntile.starpu.wait_for_all()
    A.unregister()
    B.unregister()
    tmp.unregister()
    # Compare results
    return np.allclose(np_B[0], np.linalg.norm(np_A.reshape([-1]), ord=2))

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

