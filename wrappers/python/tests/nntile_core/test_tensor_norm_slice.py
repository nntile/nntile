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
    shape=[2, 2, 2, 2],
    shape_tile=[2, 2, 2, 2],
    axis=0,
)

multiple_tiles = NormSliceTestParams(
    shape=[2, 2, 2, 2],
    shape_tile=[1, 1, 1, 1],
    axis=0,
)


def get_ref_value(alpha, src1, beta, src2, axis):
    # For norm_slice, the kernel treats the tensor as [m, k, n] where:
    # - k is the axis being reduced
    # - m is the product of all dimensions before the axis
    # - n is the product of all dimensions after the axis

    if axis == 0:
        # For axis=0 on 4D tensor [a,b,c,d]:
        # - k = a (axis being reduced)
        # - m = b*c (product of dimensions before axis)
        # - n = d (product of dimensions after axis)
        # - Result shape is [m, n] = [b*c, d] = [b, c, d]

        a, b, c, d = src1.shape
        result = np.zeros((b, c, d), dtype=src1.dtype)

        # The kernel treats the input as [m, k, n] = [b*c, a, d]
        # For each position (i2, i1) where i2 < d and i1 < b*c:
        for i2 in range(d):  # i2 corresponds to the last dimension (d)
            for i1 in range(b * c):  # i1 corresponds to the flattened b*c
                # Compute norm over k = a elements for this position
                norm_sq = 0.0
                for i0 in range(a):  # i0 corresponds to first dimension
                    # Map back to original indices for src1[a,b,c,d]
                    orig_a = i0
                    orig_b = i1 // c  # i1 // 2
                    orig_c = i1 % c   # i1 % 2
                    orig_d = i2

                    val = src1[orig_a, orig_b, orig_c, orig_d]
                    norm_sq += val * val

                fiber_norm = np.sqrt(norm_sq)

                # Map i1 back to (b, c) indices for result[b,c,d]
                b_idx = i1 // c
                c_idx = i1 % c

                result[b_idx, c_idx, i2] = np.hypot(
                    alpha * fiber_norm, beta * src2[b_idx, c_idx, i2]
                )
    else:
        # For other axes, implement accordingly
        raise NotImplementedError("Only axis=0 implemented in reference")

    return result


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
    # For norm_slice with axis=0 on 4D tensor [a,b,c,d]:
    # - The result shape is [b,c,d] (src1.shape[1:])
    # - m = b*c (product of dimensions before axis)
    # - n = d (product of dimensions after axis)
    b, c, d = src_shape[1:]  # Unpack dimensions after axis 0
    src_shape_dst = [b, c, d]  # Result shape is src1.shape[1:]
    src_tile_dst = [src_tile[1], src_tile[2], src_tile[3]]

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
