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
from typing import Sequence
import nntile
import pytest

# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {
        np.float32: nntile.tensor.Tensor_fp32,
        np.float64: nntile.tensor.Tensor_fp64
}
# Define mapping between tested function and numpy type
conv2d = {
    np.float32: nntile.nntile_core.tensor.conv2d_fp32,
    np.float64: nntile.nntile_core.tensor.conv2d_fp64,
}


@pytest.mark.parametrize('dtype', conv2d.keys())
@pytest.mark.parametrize('shape_A, shape_A_tile, shape_B, shape_B_tile, '
        'shape_C_tile', [
    [[8, 8], [8, 8], [8, 8], [8, 8], [1, 1]],
    [[8, 8], [3, 5], [8, 8], [3, 5], [1, 1]],
    [[3, 5], [1, 1], [2, 2], [1, 1], [1, 1]]
])
@pytest.mark.parametrize('in_channels, in_channels_tile', [
    [3, 3],
    [3, 1]
])
@pytest.mark.parametrize('out_channels, out_channels_tile', [
    [3, 3],
    [3, 1]
])
@pytest.mark.parametrize('batch, batch_tile', [
    [1, 1],
    [3, 3],
    [3, 1]
])
@pytest.mark.parametrize('padding', [[0, 0], [2, 1]])
def test_conv2d(
    starpu_simple,
    numpy_rng,
    dtype,
    shape_A,
    shape_A_tile,
    shape_B,
    shape_B_tile,
    shape_C_tile,
    in_channels,
    in_channels_tile,
    out_channels,
    out_channels_tile,
    batch,
    batch_tile,
    padding,
):
    next_tag = 0

    shape = [*shape_A, in_channels, batch]
    tile_shape = [*shape_A_tile, in_channels_tile, batch_tile]
    traits = nntile.tensor.TensorTraits(shape, tile_shape)
    mpi_distr = [0] * traits.grid.nelems
    A = Tensor[dtype](traits, mpi_distr, next_tag)
    src_A = np.array(
            numpy_rng.standard_normal(shape, dtype=dtype),
            dtype=dtype,
            order="F"
    )
    next_tag = A.next_tag

    shape = [*shape_B, out_channels, in_channels]
    tile_shape = [*shape_B_tile, out_channels_tile, in_channels_tile]
    traits = nntile.tensor.TensorTraits(shape, tile_shape)
    mpi_distr = [0] * traits.grid.nelems
    B = Tensor[dtype](traits, mpi_distr, next_tag)
    src_B = np.array(
            numpy_rng.standard_normal(shape, dtype=dtype),
            dtype=dtype,
            order="F"
    )
    next_tag = B.next_tag

    shape = [
        shape_A[0] + shape_B[0] - 1 - 2 * padding[0],
        shape_A[1] + shape_B[1] - 1 - 2 * padding[1],
        out_channels,
        batch,
    ]
    tile_shape = [*shape_C_tile, out_channels_tile, batch_tile]
    traits = nntile.tensor.TensorTraits(shape, tile_shape)
    mpi_distr = [0] * traits.grid.nelems
    C = Tensor[dtype](traits, mpi_distr, next_tag)
    src_C = np.array(
            numpy_rng.standard_normal(shape, dtype=dtype),
            dtype=dtype,
            order="F"
    )
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
            diff = np.linalg.norm(result - value)
            norm = np.linalg.norm(value)
            assert diff <= 1e-4 * norm
