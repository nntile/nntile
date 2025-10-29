# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_norm.py
# Test for tensor::norm<T> Python wrapper
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
norm = {
    np.float32: nntile.nntile_core.tensor.norm_fp32,
    np.float64: nntile.nntile_core.tensor.norm_fp64
}


@dataclass
class NormTestParams:
    shape: list[int]
    shape_tile: list[int]


single_tile = NormTestParams(
    shape=[5],
    shape_tile=[5]
)

multiple_tiles = NormTestParams(
    shape=[10, 20],
    shape_tile=[5, 10]
)


def get_ref_value(alpha, src, beta, dst_init):
    # Compute Euclidean norm of all elements in src
    x = alpha * np.linalg.norm(src.flatten())
    y = beta * dst_init
    return np.hypot(x, y)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
@pytest.mark.parametrize('alpha', [1.0, -1.0, 2.5])
@pytest.mark.parametrize('beta', [0.0, 1.0, -0.5])
def test_norm_async(context, dtype, params, alpha, beta):
    src_shape = params.shape
    src_tile = params.shape_tile
    dst_shape = []  # scalar tensor (empty shape)
    dst_tile = []   # scalar tensor (empty tile shape)

    rng = np.random.default_rng(42)

    # data generation
    traits_src = nntile.tensor.TensorTraits(src_shape, src_tile)
    src = Tensor[dtype](traits_src)
    np_src = rng.random(src_shape).astype(dtype, order='F')
    src.from_array(np_src)

    traits_dst = nntile.tensor.TensorTraits(dst_shape, dst_tile)
    dst = Tensor[dtype](traits_dst)
    dst_init_val = rng.random()
    np_dst_init = np.array([dst_init_val], dtype=dtype)
    dst.from_array(np_dst_init)

    # actual calculations
    norm[dtype](alpha, src, beta, dst)

    # reference value
    ref = get_ref_value(alpha, np_src, beta, dst_init_val)
    nntile_result = nntc.to_numpy(dst)

    assert np.allclose(nntile_result.flatten(), [ref], rtol=1e-5, atol=1e-6)
