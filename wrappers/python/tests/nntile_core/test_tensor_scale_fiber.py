# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_scale_fiber.py
# Test for tensor::scale_fiber<T> Python wrapper
#
# @version 1.1.0

import pytest
import numpy as np
from nntile import nntile_core


def test_scale_fiber_async_fp32(context):
    """Test scale_fiber_async for fp32"""
    # Create test data
    m, n, k, batch = 2, 3, 4, 5
    alpha = 2.0
    
    # Create source tensor (fiber)
    src_shape = [k] + [1] * batch
    src_tensor = nntile_core.Tensor_fp32(src_shape, context)
    src_array = np.random.randn(*src_shape).astype(np.float32)
    src_tensor.from_array(src_array)
    
    # Create destination tensor
    dst_shape = [m] + [1] * k + [1] * n + [1] * batch
    dst_tensor = nntile_core.Tensor_fp32(dst_shape, context)
    dst_array = np.random.randn(*dst_shape).astype(np.float32)
    dst_tensor.from_array(dst_array)
    
    # Compute reference result
    dst_ref = np.zeros_like(dst_array)
    for b in range(batch):
        for i2 in range(k):
            src_val = alpha * src_array[i2 + b * k]
            for i1 in range(n):
                for i0 in range(m):
                    idx = ((i1 + b * n) * k + i2) * m + i0
                    dst_ref[idx] = src_val
    
    # Call scale_fiber_async
    nntile_core.scale_fiber_async_fp32(alpha, src_tensor, dst_tensor, 0, batch)
    
    # Wait for completion
    nntile_core.starpu_wait_for_all()
    
    # Get result
    dst_result = dst_tensor.to_array()
    
    # Check result
    np.testing.assert_allclose(dst_result, dst_ref, rtol=1e-5, atol=1e-6)


def test_scale_fiber_async_fp64(context):
    """Test scale_fiber_async for fp64"""
    # Create test data
    m, n, k, batch = 2, 3, 4, 5
    alpha = 2.0
    
    # Create source tensor (fiber)
    src_shape = [k] + [1] * batch
    src_tensor = nntile_core.Tensor_fp64(src_shape, context)
    src_array = np.random.randn(*src_shape).astype(np.float64)
    src_tensor.from_array(src_array)
    
    # Create destination tensor
    dst_shape = [m] + [1] * k + [1] * n + [1] * batch
    dst_tensor = nntile_core.Tensor_fp64(dst_shape, context)
    dst_array = np.random.randn(*dst_shape).astype(np.float64)
    dst_tensor.from_array(dst_array)
    
    # Compute reference result
    dst_ref = np.zeros_like(dst_array)
    for b in range(batch):
        for i2 in range(k):
            src_val = alpha * src_array[i2 + b * k]
            for i1 in range(n):
                for i0 in range(m):
                    idx = ((i1 + b * n) * k + i2) * m + i0
                    dst_ref[idx] = src_val
    
    # Call scale_fiber_async
    nntile_core.scale_fiber_async_fp64(alpha, src_tensor, dst_tensor, 0, batch)
    
    # Wait for completion
    nntile_core.starpu_wait_for_all()
    
    # Get result
    dst_result = dst_tensor.to_array()
    
    # Check result
    np.testing.assert_allclose(dst_result, dst_ref, rtol=1e-10, atol=1e-12)


def test_scale_fiber_fp32(context):
    """Test scale_fiber for fp32"""
    # Create test data
    m, n, k, batch = 2, 3, 4, 5
    alpha = 2.0
    
    # Create source tensor (fiber)
    src_shape = [k] + [1] * batch
    src_tensor = nntile_core.Tensor_fp32(src_shape, context)
    src_array = np.random.randn(*src_shape).astype(np.float32)
    src_tensor.from_array(src_array)
    
    # Create destination tensor
    dst_shape = [m] + [1] * k + [1] * n + [1] * batch
    dst_tensor = nntile_core.Tensor_fp32(dst_shape, context)
    dst_array = np.random.randn(*dst_shape).astype(np.float32)
    dst_tensor.from_array(dst_array)
    
    # Compute reference result
    dst_ref = np.zeros_like(dst_array)
    for b in range(batch):
        for i2 in range(k):
            src_val = alpha * src_array[i2 + b * k]
            for i1 in range(n):
                for i0 in range(m):
                    idx = ((i1 + b * n) * k + i2) * m + i0
                    dst_ref[idx] = src_val
    
    # Call scale_fiber
    nntile_core.scale_fiber_fp32(alpha, src_tensor, dst_tensor, 0, batch)
    
    # Get result
    dst_result = dst_tensor.to_array()
    
    # Check result
    np.testing.assert_allclose(dst_result, dst_ref, rtol=1e-5, atol=1e-6)


def test_scale_fiber_fp64(context):
    """Test scale_fiber for fp64"""
    # Create test data
    m, n, k, batch = 2, 3, 4, 5
    alpha = 2.0
    
    # Create source tensor (fiber)
    src_shape = [k] + [1] * batch
    src_tensor = nntile_core.Tensor_fp64(src_shape, context)
    src_array = np.random.randn(*src_shape).astype(np.float64)
    src_tensor.from_array(src_array)
    
    # Create destination tensor
    dst_shape = [m] + [1] * k + [1] * n + [1] * batch
    dst_tensor = nntile_core.Tensor_fp64(dst_shape, context)
    dst_array = np.random.randn(*dst_shape).astype(np.float64)
    dst_tensor.from_array(dst_array)
    
    # Compute reference result
    dst_ref = np.zeros_like(dst_array)
    for b in range(batch):
        for i2 in range(k):
            src_val = alpha * src_array[i2 + b * k]
            for i1 in range(n):
                for i0 in range(m):
                    idx = ((i1 + b * n) * k + i2) * m + i0
                    dst_ref[idx] = src_val
    
    # Call scale_fiber
    nntile_core.scale_fiber_fp64(alpha, src_tensor, dst_tensor, 0, batch)
    
    # Get result
    dst_result = dst_tensor.to_array()
    
    # Check result
    np.testing.assert_allclose(dst_result, dst_ref, rtol=1e-10, atol=1e-12)