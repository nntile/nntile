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

import numpy as np
import pytest

import nntile

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


def get_ref_value(beta, dst, alpha, src):
    """
    Hardcored emulation kernel of norm fiber using numpy,
    only for this test.
    """
    nntile_shape = [src.shape[0], src.shape[1], src.shape[2] * src.shape[3]]
    src = src.reshape(nntile_shape)[:, :, :, None]
    # M, K, N, batch
    _, K, _, batch = src.shape
    dst = dst.reshape([K, batch])
    for k in range(K):
        for b in range(batch):
            norm_value = np.linalg.norm(src[:, k, :, b])
            dst[k, b] = np.hypot(beta * dst[k, b], alpha * norm_value)
    return dst


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('input_shape', [
    [3, 5, 20, 20],
    [7, 5, 21, 21]
])
def test_norm_fiber_async(context, dtype, inplace, input_shape):
    # Describe single-tile tensor
    alpha = float(1.0)
    beta = float(0.0)
    shape_A = np.array(input_shape)
    shape_B = shape_A[1:2]
    shape_C = shape_A[1:2]

    rng = np.random.default_rng(0)

    # data generation
    traits_A = nntile.tensor.TensorTraits(shape_A, shape_A)
    A = Tensor[dtype](traits_A)
    np_A = rng.random(shape_A).astype(dtype, order='F')
    A.from_array(np_A)

    traits_B = nntile.tensor.TensorTraits(shape_B, shape_B)
    B = Tensor[dtype](traits_B)
    np_B = rng.random(shape_B).astype(dtype, order='F')
    B.from_array(np_B)

    traits_C = nntile.tensor.TensorTraits(shape_C, shape_C)
    C = Tensor[dtype](traits_C)
    np_C = rng.random(shape_C).astype(dtype, order='F')
    C.from_array(np_C)

    # acutal calculations
    if inplace:
        norm_fiber_inplace[dtype](alpha, A, beta, B, 1, 0, 0)
    else:
        norm_fiber[dtype](alpha, A, beta, C, B, 1, 0, 0)
    B.to_array(np_B)
    nntile.starpu.wait_for_all()

    # reference value
    ref = get_ref_value(beta, np_B, alpha, np_A)

    A.unregister()
    B.unregister()
    C.unregister()
    assert np.allclose(np_B.flatten(), ref.flatten())
