# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_scale_slice.py
# Test for tensor::scale_slice<T> Python wrapper
#
# @version 1.1.0

import numpy as np
import pytest

import nntile

# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
          np.float64: nntile.tensor.Tensor_fp64}

# Define mapping between tested function and numpy type
scale_slice = {
    np.float32: nntile.nntile_core.tensor.scale_slice_fp32,
    np.float64: nntile.nntile_core.tensor.scale_slice_fp64
}

add_slice_inplace = {
    np.float32: nntile.nntile_core.tensor.add_slice_inplace_fp32,
    np.float64: nntile.nntile_core.tensor.add_slice_inplace_fp64
}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_scale_slice(context, dtype):
    # Describe single-tile tensor
    dst_shape = [2, 3, 4]
    src_shape = []
    ndim = len(dst_shape)
    for i in range(ndim):
        src_shape.append(dst_shape[:i] + dst_shape[i + 1:])
    dst_traits = nntile.tensor.TensorTraits(dst_shape, dst_shape)
    src_traits = []
    for i in range(ndim):
        src_traits.append(nntile.tensor.TensorTraits(
            src_shape[i], src_shape[i]))
    # Tensor objects
    dst = Tensor[dtype](dst_traits)
    dst_ref = Tensor[dtype](dst_traits)
    src = []
    for i in range(ndim):
        src.append(Tensor[dtype](src_traits[i]))
    # Set initial values of tensors
    rng = np.random.default_rng(42)
    np_src = []
    for i in range(ndim):
        rand_src = rng.standard_normal(src_shape[i])
        np_src.append(np.array(rand_src, dtype=dtype, order='F'))
        src[i].from_array(np_src[-1])
    np_dst = np.zeros(dst_shape, dtype=dtype, order='F')
    np_dst_ref = np.zeros(dst_shape, dtype=dtype, order='F')
    # Check result along each axis
    alpha = 2.5
    for i in range(ndim):
        # Test scale_slice
        scale_slice[dtype](alpha, src[i], dst, i)
        dst.to_array(np_dst)
        # Test add_slice_inplace with beta=0 for reference
        add_slice_inplace[dtype](alpha, src[i], 0.0, dst_ref, i)
        dst_ref.to_array(np_dst_ref)
        nntile.starpu.wait_for_all()
        # Verify that scale_slice gives same result as
        # add_slice_inplace with beta=0
        assert np.allclose(np_dst, np_dst_ref)
        # Also verify against numpy reference
        np_expected = alpha * np.expand_dims(np_src[i], axis=i)
        np_expected = np.repeat(np_expected, dst_shape[i], axis=i)
        assert np.allclose(np_dst, np_expected)
        src[i].unregister()
    dst.unregister()
    dst_ref.unregister()
