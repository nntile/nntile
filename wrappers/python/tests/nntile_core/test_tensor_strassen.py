# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_strassen.py
# Test for tensor::strassen<T> Python wrapper
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-12-18

# All necesary imports
import nntile
import numpy as np
import itertools

# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32, np.float64: nntile.tensor.Tensor_fp64}
# Define mapping between tested function and numpy type
strassen = {
    np.float32: nntile.nntile_core.tensor.strassen_fp32,
    np.float64: nntile.nntile_core.tensor.strassen_fp64,
}


# Helper function returns bool value true if test passes
def helper(dtype, tA, tB, matrix_shape, shared_size, tile_size, alpha, beta, batch=3):
    # Describe single-tile tensor, located at node 0
    mpi_distr = [0]
    next_tag = 0
    tile_shape = [tile_size, tile_size, 1]

    if tA == nntile.notrans:
        shape = [matrix_shape[0], shared_size, batch]
    else:
        shape = [shared_size, matrix_shape[0], batch]
    traits = nntile.tensor.TensorTraits(shape, tile_shape)
    mpi_distr = [0] * traits.grid.nelems
    A = Tensor[dtype](traits, mpi_distr, next_tag)
    src_A = np.array(np.random.randn(*shape), dtype=dtype, order="F")
    next_tag = A.next_tag

    if tB == nntile.notrans:
        shape = [shared_size, matrix_shape[1], batch]
    else:
        shape = [matrix_shape[1], shared_size, batch]
    traits = nntile.tensor.TensorTraits(shape, tile_shape)
    mpi_distr = [0] * traits.grid.nelems
    B = Tensor[dtype](traits, mpi_distr, next_tag)
    src_B = np.array(np.random.randn(*shape), dtype=dtype, order="F")
    next_tag = B.next_tag

    shape = [matrix_shape[0], matrix_shape[1], batch]
    traits = nntile.tensor.TensorTraits(shape, tile_shape)
    mpi_distr = [0] * traits.grid.nelems
    C = Tensor[dtype](traits, mpi_distr, next_tag)
    src_C = np.array(np.random.randn(*shape), dtype=dtype, order="F")
    dst_C = np.zeros_like(src_C, dtype=dtype, order="F")

    # Set initial values of tensors
    A.from_array(src_A)
    B.from_array(src_B)
    C.from_array(src_C)

    C.to_array(dst_C)
    strassen[dtype](alpha, tA, A, tB, B, beta, C, 1, 1, 0)
    C.to_array(dst_C)

    nntile.starpu.wait_for_all()
    A.unregister()
    B.unregister()
    C.unregister()

    # Check results
    for i in range(batch):
        # Get result in numpy
        termA = src_A[:, :, i] if tA == nntile.notrans else src_A[:, :, i].T
        termB = src_B[:, :, i] if tB == nntile.notrans else src_B[:, :, i].T
        src_C[:, :, i] = beta * src_C[:, :, i] + alpha * (termA @ termB)
        # Check if results are almost equal
        if (
            np.linalg.norm(dst_C[:, :, i] - src_C[:, :, i])
            / np.linalg.norm(src_C[:, :, i])
            > 1e-4
        ):
            return False
    return True


# Repeat tests for different configurations
def tests():
    trans = [nntile.notrans, nntile.trans]
    matrix_sizes = [[8, 8], [8, 4], [4, 8], [6, 4], [4, 6], [6, 6]]
    shared_sizes = range(2, 10, 2)
    tile_sizes = [2]
    ab = [-0.5, -0.33, 0.33, 0.5]
    args_sets = itertools.product(
        dtypes, trans, trans, matrix_sizes, shared_sizes, tile_sizes, ab, ab
    )
    for args in args_sets:
        assert helper(*args)


if __name__ == "__main__":
    tests()
