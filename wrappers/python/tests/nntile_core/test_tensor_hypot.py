# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_hypot.py
# Test for tensor::hypot<T> Python wrapper
#
# @version 1.1.0

import numpy as np
import pytest

import nntile

# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
          np.float64: nntile.tensor.Tensor_fp64}
# Define mapping between tested function and numpy type
hypot = {np.float32: nntile.nntile_core.tensor.hypot_fp32,
         np.float64: nntile.nntile_core.tensor.hypot_fp64}


def hypot_numpy(alpha, src1, beta, src2):
    """
    Reference implementation of hypot operation using numpy
    Computes: dst[i] = hypot(alpha * src1[i], beta * src2[i])
    """
    scaled_src1 = alpha * src1
    scaled_src2 = beta * src2

    # Handle edge cases where alpha or beta is zero
    result = np.zeros_like(scaled_src1)

    # Get reference result
    result = np.sqrt(scaled_src1**2 + scaled_src2**2)

    return result


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('alpha', [0.0, 1.0, -1.5, 2.0])
@pytest.mark.parametrize('beta', [0.0, 1.0, -1.5, 2.0])
def test_hypot(context, dtype, alpha, beta):
    # Describe single-tile tensor, located at node 0
    shape = [2, 2]
    traits = nntile.tensor.TensorTraits(shape, shape)

    # Tensor objects
    src1 = Tensor[dtype](traits, [0])
    src2 = Tensor[dtype](traits, [0])
    dst = Tensor[dtype](traits, [0])

    # Generate input data
    rand1 = np.random.default_rng(42).standard_normal(shape)
    rand2 = np.random.default_rng(123).standard_normal(shape)
    src1_data = np.array(rand1, dtype=dtype, order='F')
    src2_data = np.array(rand2, dtype=dtype, order='F')
    dst_data = np.zeros_like(src1_data)

    src1.from_array(src1_data)
    src2.from_array(src2_data)
    dst.from_array(dst_data)

    # Apply hypot
    hypot[dtype](alpha, src1, beta, src2, dst)
    dst.to_array(dst_data)
    nntile.starpu.wait_for_all()

    # Cleanup
    src1.unregister()
    src2.unregister()
    dst.unregister()

    # Compute reference result
    dst_ref = hypot_numpy(alpha, src1_data, beta, src2_data)

    # Compare (use relative tolerance for floating point)
    if dtype == np.float32:
        rtol = 1e-6
        atol = 1e-8
    else:  # float64
        rtol = 1e-12
        atol = 1e-15

    assert np.allclose(dst_ref, dst_data, rtol=rtol, atol=atol)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_hypot_edge_cases(context, dtype):
    """Test edge cases for hypot operation"""
    shape = [3]
    traits = nntile.tensor.TensorTraits(shape, shape)

    src1 = Tensor[dtype](traits, [0])
    src2 = Tensor[dtype](traits, [0])
    dst = Tensor[dtype](traits, [0])

    # Test case: alpha=0, beta=0
    src1_data = np.array([1.0, 2.0, 3.0], dtype=dtype, order='F')
    src2_data = np.array([4.0, 5.0, 6.0], dtype=dtype, order='F')
    dst_data = np.zeros_like(src1_data)

    src1.from_array(src1_data)
    src2.from_array(src2_data)
    dst.from_array(dst_data)

    hypot[dtype](0.0, src1, 0.0, src2, dst)
    dst.to_array(dst_data)
    nntile.starpu.wait_for_all()

    # Should result in zeros
    assert np.allclose(dst_data, np.zeros_like(dst_data))

    # Cleanup
    src1.unregister()
    src2.unregister()
    dst.unregister()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_hypot_async(context, dtype):
    """Test async version of hypot"""
    shape = [2, 3]
    traits = nntile.tensor.TensorTraits(shape, shape)

    src1 = Tensor[dtype](traits, [0])
    src2 = Tensor[dtype](traits, [0])
    dst = Tensor[dtype](traits, [0])

    # Generate input data
    rand1 = np.random.default_rng(42).standard_normal(shape)
    rand2 = np.random.default_rng(123).standard_normal(shape)
    src1_data = np.array(rand1, dtype=dtype, order='F')
    src2_data = np.array(rand2, dtype=dtype, order='F')
    dst_data = np.zeros_like(src1_data)

    src1.from_array(src1_data)
    src2.from_array(src2_data)
    dst.from_array(dst_data)

    # Apply hypot async
    from nntile.nntile_core import tensor as core_tensor
    hypot_async_func = {np.float32: core_tensor.hypot_async_fp32,
                       np.float64: core_tensor.hypot_async_fp64}
    hypot_async_func[dtype](1.5, src1, -0.5, src2, dst)
    nntile.starpu.wait_for_all()
    dst.to_array(dst_data)

    # Cleanup
    src1.unregister()
    src2.unregister()
    dst.unregister()

    # Compute reference result
    dst_ref = hypot_numpy(1.5, src1_data, -0.5, src2_data)

    # Compare
    if dtype == np.float32:
        rtol = 1e-6
        atol = 1e-8
    else:  # float64
        rtol = 1e-12
        atol = 1e-15

    assert np.allclose(dst_ref, dst_data, rtol=rtol, atol=atol)
