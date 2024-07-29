# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_conv2d.py
# Test for tensor::conv2d<T> Python wrapper
#
# @version 1.0.0

import numpy as np
import scipy

# All necesary imports
import nntile

# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32, np.float64: nntile.tensor.Tensor_fp64}
# Define mapping between tested function and numpy type
conv2d = {
    np.float32: nntile.nntile_core.tensor.conv2d_fp32,
    np.float64: nntile.nntile_core.tensor.conv2d_fp64,
}


# Helper function returns bool value true if test passes
def helper(
    dtype,
    shape_A,
    shape_B,
    tile_shape_A,
    tile_shape_B,
    tile_shape_C,
    in_channels,
    out_channels,
    batch,
    padding,
):
    next_tag = 0

    shape = [*shape_A, in_channels, batch]
    traits = nntile.tensor.TensorTraits(shape, tile_shape_A)
    mpi_distr = [0] * traits.grid.nelems
    A = Tensor[dtype](traits, mpi_distr, next_tag)
    src_A = np.array(np.random.randn(*shape), dtype=dtype, order="F")
    next_tag = A.next_tag

    shape = [*shape_B, out_channels, in_channels]
    traits = nntile.tensor.TensorTraits(shape, tile_shape_B)
    mpi_distr = [0] * traits.grid.nelems
    B = Tensor[dtype](traits, mpi_distr, next_tag)
    src_B = np.array(np.random.randn(*shape), dtype=dtype, order="F")
    next_tag = B.next_tag

    shape = [
        shape_A[0] + shape_B[0] - 1 - 2 * padding[0],
        shape_A[1] + shape_B[1] - 1 - 2 * padding[1],
        out_channels,
        batch,
    ]
    traits = nntile.tensor.TensorTraits(shape, tile_shape_C)
    mpi_distr = [0] * traits.grid.nelems
    C = Tensor[dtype](traits, mpi_distr, next_tag)
    src_C = np.array(np.random.randn(*shape), dtype=dtype, order="F")
    dst_C = np.zeros_like(src_C, dtype=dtype, order="F")

    # Set initial values of tensors
    A.from_array(src_A)
    B.from_array(src_B)
    C.from_array(src_C)

    conv2d[dtype](A, B, C, padding[0], padding[1])
    C.to_array(dst_C)

    nntile.starpu.wait_for_all()
    A.unregister()
    B.unregister()
    C.unregister()

    # Check results
    for b in range(batch):
        for oc in range(out_channels):
            # Get result in numpy
            src_C[..., oc, b] = 0
            for ic in range(in_channels):
                termA = src_A[..., ic, b]
                if padding[0] != 0:
                    termA = termA[padding[0] : -padding[0], ...]
                if padding[1] != 0:
                    termA = termA[:, padding[1] : -padding[1], ...]
                termB = src_B[..., oc, ic]
                src_C[..., oc, b] += scipy.signal.convolve2d(termA, termB)
            # Check if results are almost equal
            value = src_C[..., oc, b]
            result = dst_C[..., oc, b]
            if np.linalg.norm(result - value) / np.linalg.norm(value) > 1e-4:
                return False
    return True


# Repeat tests for different configurations
def tests():
    for dtype in conv2d.keys():
        # Basic test
        assert helper(
            dtype=dtype,
            shape_A=[8, 8],
            shape_B=[8, 8],
            tile_shape_A=[8, 8, 1, 1],
            tile_shape_B=[8, 8, 1, 1],
            tile_shape_C=[15, 15, 1, 1],
            in_channels=1,
            out_channels=1,
            batch=1,
            padding=[0, 0],
        )
        # Different matrix sizes
        assert helper(
            dtype=dtype,
            shape_A=[3, 5],
            shape_B=[7, 11],
            tile_shape_A=[3, 5, 1, 1],
            tile_shape_B=[7, 11, 1, 1],
            tile_shape_C=[9, 15, 1, 1],
            in_channels=1,
            out_channels=1,
            batch=1,
            padding=[0, 0],
        )
        # With padding
        assert helper(
            dtype=dtype,
            shape_A=[3, 5],
            shape_B=[7, 11],
            tile_shape_A=[3, 5, 1, 1],
            tile_shape_B=[7, 11, 1, 1],
            tile_shape_C=[8, 13, 1, 1],
            in_channels=1,
            out_channels=1,
            batch=1,
            padding=[1, 2],
        )
        # With smaller tiles
        assert helper(
            dtype=dtype,
            shape_A=[3, 5],
            shape_B=[7, 11],
            tile_shape_A=[2, 2, 1, 1],
            tile_shape_B=[3, 3, 1, 1],
            tile_shape_C=[4, 4, 1, 1],
            in_channels=1,
            out_channels=1,
            batch=1,
            padding=[1, 2],
        )
        # With in/out channels
        assert helper(
            dtype=dtype,
            shape_A=[3, 5],
            shape_B=[7, 11],
            tile_shape_A=[2, 2, 1, 1],
            tile_shape_B=[3, 3, 1, 1],
            tile_shape_C=[4, 4, 1, 1],
            in_channels=4,
            out_channels=5,
            batch=1,
            padding=[1, 2],
        )
        # With batch
        assert helper(
            dtype=dtype,
            shape_A=[3, 5],
            shape_B=[7, 11],
            tile_shape_A=[2, 2, 1, 1],
            tile_shape_B=[3, 3, 1, 1],
            tile_shape_C=[4, 4, 1, 1],
            in_channels=4,
            out_channels=5,
            batch=3,
            padding=[1, 2],
        )
        # With tile batches
        assert helper(
            dtype=dtype,
            shape_A=[3, 5],
            shape_B=[7, 11],
            tile_shape_A=[2, 2, 2, 2],
            tile_shape_B=[3, 3, 2, 2],
            tile_shape_C=[4, 4, 2, 2],
            in_channels=4,
            out_channels=5,
            batch=3,
            padding=[1, 2],
        )


if __name__ == "__main__":
    tests()
