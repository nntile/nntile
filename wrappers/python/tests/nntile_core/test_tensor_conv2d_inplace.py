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
# @version 1.1.0

import numpy as np
import pytest
import torch

import nntile

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
conv2d_inplace = {
    np.float32: nntile.nntile_core.tensor.conv2d_inplace_fp32,
    np.float64: nntile.nntile_core.tensor.conv2d_inplace_fp64,
}


@pytest.mark.parametrize('dtype', conv2d_inplace.keys())
@pytest.mark.parametrize('shape_X, shape_X_tile, shape_W, shape_Y_tile', [
    [[2, 3], [2, 3], [2, 3], [1, 1]],
    [[2, 3], [1, 1], [2, 3], [1, 1]],
    [[8, 10], [2, 4], [3, 3], [3, 2]],
    [[28, 28], [4, 4], [5, 5], [4, 4]]
])
@pytest.mark.parametrize('in_channels', [1, 3])
@pytest.mark.parametrize('out_channels', [1, 3])
@pytest.mark.parametrize('batch, batch_tile', [
    [1, 1],
    [4, 2]
])
@pytest.mark.parametrize('padding', [[0, 0], [2, 1]])
@pytest.mark.parametrize('stride', [[1, 1], [2, 3]])
@pytest.mark.parametrize('dilation', [[1, 1], [2, 2]])
def test_conv2d(
    starpu_simple,
    numpy_rng,
    dtype,
    shape_X,
    shape_X_tile,
    shape_W,
    shape_Y_tile,
    in_channels,
    out_channels,
    batch,
    batch_tile,
    padding,
    stride,
    dilation
):
    # Get output shape
    shape_Y = [
        (shape_X[0] - dilation[0] * (shape_W[0] - 1) - 1 + 2 * padding[0])
            // stride[0] + 1,
        (shape_X[1] - dilation[1] * (shape_W[1] - 1) - 1 + 2 * padding[1])
            // stride[1] + 1,
        out_channels,
        batch,
    ]
    # Pass test with unappropriate shapes
    if shape_Y[0] <= 0 or shape_Y[1] <= 0:
        return
    next_tag = 0

    shape = [*shape_X, in_channels, batch]
    tile_shape = [*shape_X_tile, in_channels, batch_tile]
    traits = nntile.tensor.TensorTraits(shape, tile_shape)
    mpi_distr = [0] * traits.grid.nelems
    X = Tensor[dtype](traits, mpi_distr, next_tag)
    src_X = np.array(
            numpy_rng.standard_normal(shape, dtype=dtype),
            dtype=dtype,
            order="F"
    )
    next_tag = X.next_tag

    shape = [*shape_W, in_channels, out_channels]
    traits = nntile.tensor.TensorTraits(shape, shape)
    mpi_distr = [0] * traits.grid.nelems
    W = Tensor[dtype](traits, mpi_distr, next_tag)
    src_W = np.array(
            numpy_rng.standard_normal(shape, dtype=dtype),
            dtype=dtype,
            order="F"
    )
    next_tag = W.next_tag

    shape = shape_Y
    tile_shape = [*shape_Y_tile, out_channels, batch_tile]
    traits = nntile.tensor.TensorTraits(shape, tile_shape)
    mpi_distr = [0] * traits.grid.nelems
    Y = Tensor[dtype](traits, mpi_distr, next_tag)
    src_Y = np.array(
            numpy_rng.standard_normal(shape, dtype=dtype),
            dtype=dtype,
            order="F"
    )
    dst_Y = np.zeros_like(src_Y, dtype=dtype, order="F")

    # Set initial values of tensors
    X.from_array(src_X)
    W.from_array(src_W)
    Y.from_array(src_Y)

    conv2d_inplace[dtype](1.0, X, W, 0.0, Y, padding, stride, dilation)
    Y.to_array(dst_Y)

    nntile.starpu.wait_for_all()
    X.unregister()
    W.unregister()
    Y.unregister()

    # Check results
    conv_torch = torch.nn.Conv2d(in_channels, out_channels,
            kernel_size=reversed(shape_W), padding=reversed(padding),
            stride=reversed(stride), dilation=reversed(dilation), bias=False)
    conv_torch.weight.data = torch.Tensor(src_W.T)
    Y_torch = conv_torch(torch.Tensor(src_X.T))

    diff = np.linalg.norm(dst_Y - Y_torch.detach().numpy().T)
    norm = np.linalg.norm(Y_torch.detach().numpy())
    assert diff <= 1e-4 * norm
