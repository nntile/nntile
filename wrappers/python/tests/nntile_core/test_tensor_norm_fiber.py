# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_norm_fiber.py
# Test for tensor::norm_fiber<T> Python wrapper
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
norm_fiber = {
    np.float32: nntile.nntile_core.tensor.norm_fiber_async_fp32,
    np.float64: nntile.nntile_core.tensor.norm_fiber_async_fp64
}

norm_fiber_inplace = {
    np.float32: nntile.nntile_core.tensor.norm_fiber_inplace_async_fp32,
    np.float64: nntile.nntile_core.tensor.norm_fiber_inplace_async_fp64
}


@dataclass
class NormFiberTestParams:
    shape: list[int]
    shape_tile: list[int]
    axis: int
    batch_ndim: int


single_tile = NormFiberTestParams(
    shape=[2, 2, 2, 2],
    shape_tile=[2, 2, 2, 2],
    axis=0,
    batch_ndim=0
)

multiple_tiles = NormFiberTestParams(
    shape=[2, 2, 2, 2],
    shape_tile=[1, 1, 1, 1],
    axis=0,
    batch_ndim=0
)


def get_ref_value(alpha, src1, beta, src2, axis):
    # Get norm of src1 into fiber along axis
    tmp1 = src1.copy()
    # Get norm of src1 along axes 0..axis-1
    for _ in range(axis):
        tmp1 = np.linalg.norm(tmp1, axis=0)
    # Get norm of src1 along axes axis+1..src2.ndim-1
    for _ in range(axis, src1.ndim - src2.ndim):
        tmp1 = np.linalg.norm(tmp1, axis=1)
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
def test_norm_fiber_async(context, dtype, params):
    alpha = float(1.0)
    beta = float(-1.0)
    src1_shape = params.shape
    src1_tile = params.shape_tile
    src2_shape = [src1_shape[params.axis]] + \
        src1_shape[len(src1_shape) - params.batch_ndim:]
    src2_tile = [src1_tile[params.axis]] + \
        src1_tile[len(src1_tile) - params.batch_ndim:]
    dst_shape = src2_shape
    dst_tile = src2_tile

    rng = np.random.default_rng(0)

    # data generation
    traits_src1 = nntile.tensor.TensorTraits(src1_shape, src1_tile)
    src1 = Tensor[dtype](traits_src1)
    np_src1 = rng.random(src1_shape).astype(dtype, order='F')
    src1.from_array(np_src1)

    traits_src2 = nntile.tensor.TensorTraits(src2_shape, src2_tile)
    src2 = Tensor[dtype](traits_src2)
    np_src2 = rng.random(src2_shape).astype(dtype, order='F')
    src2.from_array(np_src2)

    traits_dst = nntile.tensor.TensorTraits(dst_shape, dst_tile)
    dst = Tensor[dtype](traits_dst)
    np_dst = rng.random(dst_shape).astype(dtype, order='F')
    dst.from_array(np_dst)

    # actual calculations
    redux = 0
    norm_fiber[dtype](
        alpha, src1, beta, src2, dst, params.axis, params.batch_ndim, redux)
    # reference value
    ref = get_ref_value(alpha, np_src1, beta, np_src2, params.axis)
    nntile_result = nntc.to_numpy(dst)
    nntile.starpu.wait_for_all()
    assert np.allclose(nntile_result.flatten(), ref.flatten())
