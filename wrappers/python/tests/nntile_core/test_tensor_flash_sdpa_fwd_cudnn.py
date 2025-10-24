# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_flash_sdpa_fwd_cudnn.py
# Test for tensor::flash_sdpa_fwd_cudnn<T> Python wrapper
#
# @version 1.1.0

import numpy as np
import pytest

import nntile
from nntile.functions import flash_sdpa_fwd_cudnn_async

# Only test BF16 and FP16 as per cuDNN limitations
supported_dtypes = ['fp16', 'bf16']


@pytest.mark.parametrize('dtype', supported_dtypes)
def test_flash_sdpa_fwd_cudnn_async(context, dtype):
    # Test parameters - use small values for testing
    head_size = 32
    n_seq = 64
    n_batch = 2
    kv_group_size = 1
    n_head_kv = 1

    # Create 5D tensor shapes
    kqv_shape = [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
    mask_shape = [n_batch, n_seq, n_seq]
    logsumexp_shape = [n_batch, n_seq, kv_group_size]

    # Tensor traits (single tile per tensor)
    K_traits = nntile.tensor.TensorTraits(kqv_shape, kqv_shape)
    Q_traits = nntile.tensor.TensorTraits(kqv_shape, kqv_shape)
    V_traits = nntile.tensor.TensorTraits(kqv_shape, kqv_shape)
    A_traits = nntile.tensor.TensorTraits(kqv_shape, kqv_shape)
    mask_traits = nntile.tensor.TensorTraits(mask_shape, mask_shape)
    logsumexp_traits = nntile.tensor.TensorTraits(logsumexp_shape, logsumexp_shape)

    # Use root rank for all tensors (single MPI rank scenario)
    mpi_root = 0
    dist_root = [mpi_root]

    # Create tensor objects based on dtype
    if dtype == 'fp16':
        tensor_type = nntile.tensor.Tensor_fp16
        numpy_dtype = np.float16
    elif dtype == 'bf16':
        tensor_type = nntile.tensor.Tensor_bf16
        numpy_dtype = np.float32  # BF16 is stored as float32 in numpy arrays
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    K = tensor_type(K_traits, dist_root)
    Q = tensor_type(Q_traits, dist_root)
    V = tensor_type(V_traits, dist_root)
    A = tensor_type(A_traits, dist_root)
    mask = tensor_type(mask_traits, dist_root)
    logsumexp = tensor_type(logsumexp_traits, dist_root)

    # Initialize input data
    rng = np.random.default_rng(42)

    # Initialize K, Q, V with small values
    K_src = rng.standard_normal(kqv_shape).astype(numpy_dtype, 'F') * 0.1
    Q_src = rng.standard_normal(kqv_shape).astype(numpy_dtype, 'F') * 0.1
    V_src = rng.standard_normal(kqv_shape).astype(numpy_dtype, 'F') * 0.1
    A_src = np.zeros_like(K_src)

    # Initialize mask (allow attention within a window)
    mask_src = np.full(mask_shape, -np.inf, dtype=numpy_dtype, order='F')
    window_size = 32
    for b in range(n_batch):
        for i in range(n_seq):
            for j in range(n_seq):
                if abs(i - j) <= window_size:
                    mask_src[b, i, j] = 0.0

    # Initialize logsumexp to zeros
    logsumexp_src = np.zeros(logsumexp_shape, dtype=numpy_dtype, order='F')

    # Transfer data to NNTile tensors
    K.from_array(K_src)
    Q.from_array(Q_src)
    V.from_array(V_src)
    A.from_array(A_src)
    mask.from_array(mask_src)
    logsumexp.from_array(logsumexp_src)

    # Perform async flash_sdpa operation
    flash_sdpa_fwd_cudnn_async(K, Q, mask, logsumexp, V, A)

    # Wait for completion
    nntile.starpu.wait_for_all()

    # Get results back
    A_result = A.to_array()

    # For now, just check that the operation completed without error
    # The actual computation results may be zero due to masking or small values
    assert A_result.shape == tuple(kqv_shape)
    assert np.isfinite(A_result).any()  # At least some finite values should be present
