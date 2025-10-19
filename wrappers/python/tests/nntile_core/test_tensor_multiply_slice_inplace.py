#!/usr/bin/env python3

import numpy as np
import nntile

# Test for tensor::multiply_slice_inplace<T> Python wrapper

def test_multiply_slice_inplace_async(context):
    # Set up types
    alpha = 1.5
    beta = 0.5
    axis = 1
    # Set up shapes
    shape_src = [2, 3]
    shape_dst = [2, 4, 3]
    # Create tensors
    src = nntile.tensor.Tensor_fp32(shape_src, dtype=np.float32)
    dst = nntile.tensor.Tensor_fp32(shape_dst, dtype=np.float32)
    # Generate random arrays
    rng = np.random.default_rng(42)
    rand_src = rng.standard_normal(shape_src).astype(np.float32)
    rand_dst = rng.standard_normal(shape_dst).astype(np.float32)
    # Set up random arrays
    src.from_array(rand_src)
    dst.from_array(rand_dst)
    # Perform multiply_slice_inplace operation
    multiply_slice_inplace_async = {
        np.float32: nntile.nntile_core.tensor.multiply_slice_inplace_async_fp32,
        np.float64: nntile.nntile_core.tensor.multiply_slice_inplace_async_fp64
    }
    multiply_slice_inplace_async[np.float32](alpha, src, beta, dst, axis)
    # Check result
    dst_arr = np.array(dst)
    expected = beta * rand_dst * alpha * rand_src[:, np.newaxis, :]
    assert np.allclose(dst_arr, expected, rtol=1e-5)

if __name__ == "__main__":
    nntile.starpu.init()
    ctx = nntile.starpu.Context()
    test_multiply_slice_inplace_async(ctx)
    nntile.starpu.shutdown()
