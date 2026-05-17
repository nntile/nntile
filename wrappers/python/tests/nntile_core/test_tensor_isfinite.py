# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_isfinite.py
# Test for tensor::isfinite<T> Python wrapper
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
    np.float64: nntile.tensor.Tensor_fp64,
    bool: nntile.tensor.Tensor_bool
}

# Define mapping between tested function and numpy type
isfinite = {
    np.float32: nntile.nntile_core.tensor.isfinite_fp32,
    np.float64: nntile.nntile_core.tensor.isfinite_fp64
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


def get_ref_value(src):
    # Compute Euclidean norm of all elements in src
    return np.any(np.isfinite(src) == False)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('first_element', [1, float('inf'), -float('inf'), float('nan')])
@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
def test_isfinite_async(context, dtype, params, first_element):
    src_shape = params.shape
    src_tile = params.shape_tile
    flag_shape = []  # scalar tensor (empty shape)
    flag_tile = []   # scalar tensor (empty tile shape)

    rng = np.random.default_rng(42)

    # data generation
    traits_src = nntile.tensor.TensorTraits(src_shape, src_tile)
    src = Tensor[dtype](traits_src)
    np_src = rng.random(src_shape).astype(dtype, order='F')
    np_src[0] = first_element
    src.from_array(np_src)

    traits_flag = nntile.tensor.TensorTraits(flag_shape, flag_tile)
    flag = Tensor[bool](traits_flag)
    flag_init_val = 0
    np_dst_init = np.array([flag_init_val], dtype=bool)
    flag.from_array(np_dst_init)

    # actual calculations
    isfinite[dtype](src, flag)

    # reference value
    ref = get_ref_value(np_src)
    nntile_result = nntc.to_numpy(flag)

    print(nntile_result, ref)

    assert np.allclose(nntile_result.flatten(), [ref])