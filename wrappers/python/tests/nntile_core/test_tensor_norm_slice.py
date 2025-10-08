# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_norm_slice.py
# Test for tensor::norm_slice<T> Python wrapper
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest

import nntile
import nntile.utils.constructors as nntc

# Define mapping between numpy and nntile types
Tensor = {
    np.float32: nntile.tensor.Tensor_fp32,
    np.float64: nntile.tensor.Tensor_fp64
}

# Define mapping between tested function and numpy type
norm_slice = {
    np.float32: nntile.nntile_core.tensor.norm_slice_async_fp32,
    np.float64: nntile.nntile_core.tensor.norm_slice_async_fp64
}

norm_slice_inplace = {
    np.float32: nntile.nntile_core.tensor.norm_slice_inplace_async_fp32,
    np.float64: nntile.nntile_core.tensor.norm_slice_inplace_async_fp64
}


@dataclass
class NormSliceTestParams:
    shape: list[int]
    shape_tile: list[int]
    axis: int


single_tile = NormSliceTestParams(
    shape=[2, 3, 4],
    shape_tile=[2, 3, 4],
    axis=1,
)

multiple_tiles = NormSliceTestParams(
    shape=[2, 3, 4],
    shape_tile=[1, 1, 1],
    axis=1,
)


def get_ref_value(alpha, src1, beta, src2, axis):
    # Get norm of src1 along axis (which should be axis=1 for 3D tensors [m, k, n])
    tmp1 = src1.copy()
    # Get norm of src1 along axis 1 (the middle axis)
    tmp1 = np.linalg.norm(tmp1, axis=1, keepdims=True)
    # Squeeze to remove the axis dimension
    tmp1 = np.squeeze(tmp1, axis=1)
    # Now shape of tmp1 must be the same as src2
    if tmp1.shape != src2.shape:
        raise ValueError("Shape of tmp1 and src2 must be the same")
    # Now we can do hypot of src2 and tmp1
    dst = np.hypot(alpha * tmp1, beta * src2)
    return dst


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
def test_norm_slice_async(context, dtype, params):
    alpha = float(1.0)
    beta = float(-1.0)
    src_shape = params.shape
    src_tile = params.shape_tile
    # For norm_slice, src has shape [m, k, n] and dst/result have shape [m, n]
    # where axis is the middle axis (1)
    src_shape_dst = [src_shape[0], src_shape[2]]
    src_tile_dst = [src_tile[0], src_tile[2]]

    rng = np.random.default_rng(0)

    # data generation
    traits_src = nntile.tensor.TensorTraits(src_shape, src_tile)
    src = Tensor[dtype](traits_src)
    np_src = rng.random(src_shape).astype(dtype, order='F')
    src.from_array(np_src)

    traits_dst = nntile.tensor.TensorTraits(src_shape_dst, src_tile_dst)
    dst = Tensor[dtype](traits_dst)
    np_dst = rng.random(src_shape_dst).astype(dtype, order='F')
    dst.from_array(np_dst)

    traits_result = nntile.tensor.TensorTraits(src_shape_dst, src_tile_dst)
    result = Tensor[dtype](traits_result)
    np_result = rng.random(src_shape_dst).astype(dtype, order='F')
    result.from_array(np_result)

    # actual calculations
    redux = 0
    norm_slice[dtype](
        alpha, src, beta, dst, result, params.axis, redux)
    # reference value
    ref = get_ref_value(alpha, np_src, beta, np_dst, params.axis)
    nntile_result = nntc.to_numpy(result)
    nntile.starpu.wait_for_all()
    assert np.allclose(nntile_result.flatten(), ref.flatten())