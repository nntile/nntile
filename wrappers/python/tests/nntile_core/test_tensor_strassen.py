# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                             (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                             (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_strassen.py
# Test for tensor::strassen<T> Python wrapper
#
# @version 1.0.0

# All necesary imports
import nntile
import numpy as np
import pytest

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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("tA", [nntile.notrans, nntile.trans])
@pytest.mark.parametrize("tB", [nntile.notrans, nntile.trans])
@pytest.mark.parametrize(
    "matrix_shape", [[8, 8], [8, 4], [4, 8], [6, 4], [4, 6], [6, 6]]
)
@pytest.mark.parametrize("shared_size", [2, 6, 12])
@pytest.mark.parametrize("tile_size", [1, 2])
@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.5])
@pytest.mark.parametrize("beta", [0.0, 1.0, -0.5])
@pytest.mark.parametrize("batches", [[1], [3], [3, 3]])
# Helper function returns bool value true if test passes
def test_strassen(
    dtype, tA, tB, matrix_shape, shared_size, tile_size, alpha, beta, batches
):
    # Describe single-tile tensor, located at node 0
    mpi_distr = [0]
    next_tag = 0
    tile_shape = [tile_size, tile_size] + [1] * len(batches)

    if tA == nntile.notrans:
        shape = [matrix_shape[0], shared_size, *batches]
    else:
        shape = [shared_size, matrix_shape[0], *batches]
    traits = nntile.tensor.TensorTraits(shape, tile_shape)
    mpi_distr = [0] * traits.grid.nelems
    A = Tensor[dtype](traits, mpi_distr, next_tag)
    src_A = np.array(np.random.randn(*shape), dtype=dtype, order="F")
    next_tag = A.next_tag

    if tB == nntile.notrans:
        shape = [shared_size, matrix_shape[1], *batches]
    else:
        shape = [matrix_shape[1], shared_size, *batches]
    traits = nntile.tensor.TensorTraits(shape, tile_shape)
    mpi_distr = [0] * traits.grid.nelems
    B = Tensor[dtype](traits, mpi_distr, next_tag)
    src_B = np.array(np.random.randn(*shape), dtype=dtype, order="F")
    next_tag = B.next_tag

    shape = [matrix_shape[0], matrix_shape[1], *batches]
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
    strassen[dtype](alpha, tA, A, tB, B, beta, C, 1, len(batches), 0)
    C.to_array(dst_C)

    nntile.starpu.wait_for_all()
    A.unregister()
    B.unregister()
    C.unregister()

    batch = np.prod(batches)
    src_A = src_A.reshape(*src_A.shape[:2], batch)
    src_B = src_B.reshape(*src_B.shape[:2], batch)
    src_C = src_C.reshape(*src_C.shape[:2], batch)
    dst_C = dst_C.reshape(*dst_C.shape[:2], batch)

    # Check results
    for i in range(batch):
        termA = src_A[:, :, i] if tA == nntile.notrans else src_A[:, :, i].T
        termB = src_B[:, :, i] if tB == nntile.notrans else src_B[:, :, i].T
        src_C[:, :, i] = beta * src_C[:, :, i] + alpha * (termA @ termB)
    # Check if results are almost equal
    result = dst_C
    value = src_C
    diff = np.linalg.norm(result - value)
    norm = np.linalg.norm(value)
    assert diff <= 1e-4 * norm
