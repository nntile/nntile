#!/usr/bin/env python3

#! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                 2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_scale_slice.py
# Tests for scale_slice operation
#
# @version 1.1.0
# */

import numpy as np
import nntile
import pytest

def test_scale_slice():
    """Test scale_slice operation"""
    # Test parameters
    m, n, k = 3, 4, 5
    alpha = 2.0
    
    # Create source tensor (2D)
    src_data = np.arange(m * n, dtype=np.float32).reshape(m, n)
    src = nntile.tensor.Tensor_fp32(src_data, [m, n], [1, m])
    
    # Create destination tensor (3D)
    dst_data = np.zeros((m, k, n), dtype=np.float32)
    dst = nntile.tensor.Tensor_fp32(dst_data, [m, k, n], [1, m, m*k])
    
    # Create reference tensor using add_slice_inplace with beta=0
    dst_ref_data = np.zeros((m, k, n), dtype=np.float32)
    dst_ref = nntile.tensor.Tensor_fp32(dst_ref_data, [m, k, n], [1, m, m*k])
    
    # Test scale_slice
    nntile.tensor.scale_slice_fp32(alpha, src, dst, 1)
    
    # Test add_slice_inplace with beta=0 for reference
    nntile.tensor.add_slice_inplace_fp32(alpha, src, 0.0, dst_ref, 1)
    
    # Check results
    np.testing.assert_array_almost_equal(dst.to_array(), dst_ref.to_array())

def test_scale_slice_async():
    """Test scale_slice_async operation"""
    # Test parameters
    m, n, k = 3, 4, 5
    alpha = 2.0
    
    # Create source tensor (2D)
    src_data = np.arange(m * n, dtype=np.float32).reshape(m, n)
    src = nntile.tensor.Tensor_fp32(src_data, [m, n], [1, m])
    
    # Create destination tensor (3D)
    dst_data = np.zeros((m, k, n), dtype=np.float32)
    dst = nntile.tensor.Tensor_fp32(dst_data, [m, k, n], [1, m, m*k])
    
    # Create reference tensor using add_slice_inplace with beta=0
    dst_ref_data = np.zeros((m, k, n), dtype=np.float32)
    dst_ref = nntile.tensor.Tensor_fp32(dst_ref_data, [m, k, n], [1, m, m*k])
    
    # Test scale_slice_async
    nntile.tensor.scale_slice_async_fp32(alpha, src, dst, 1)
    nntile.starpu.wait_for_all()
    
    # Test add_slice_inplace with beta=0 for reference
    nntile.tensor.add_slice_inplace_async_fp32(alpha, src, 0.0, dst_ref, 1)
    nntile.starpu.wait_for_all()
    
    # Check results
    np.testing.assert_array_almost_equal(dst.to_array(), dst_ref.to_array())

if __name__ == "__main__":
    test_scale_slice()
    test_scale_slice_async()
    print("All tests passed!")
