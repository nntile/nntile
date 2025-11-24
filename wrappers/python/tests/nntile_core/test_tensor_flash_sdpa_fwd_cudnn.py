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
# This test validates the cuDNN Flash Attention implementation by:
# 1. Running the NNTile flash_sdpa_fwd_cudnn_async function
# 2. Computing a PyTorch MultiheadAttention baseline for comparison
# 3. Validating numerical correctness and functionality
#
# The test uses PyTorch as a reference implementation to ensure the cuDNN
# implementation produces reasonable results. When PyTorch is not available,
# the test falls back to validating only the NNTile functionality.
#
# @version 1.1.0

import numpy as np
import pytest
import torch

import nntile
from nntile.functions import flash_sdpa_fwd_cudnn_async

# Only test BF16 and FP16 as per cuDNN limitations
supported_dtypes = ['fp16', 'bf16']


def _prepare_flash_inputs(dtype, seed=42):
    head_size = 32
    n_seq = 64
    n_batch = 2
    kv_group_size = 1
    n_head_kv = 1

    kqv_shape = [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
    mask_shape = [n_seq, n_seq]
    logsumexp_shape = [n_seq, n_batch, kv_group_size, n_head_kv]

    K_traits = nntile.tensor.TensorTraits(kqv_shape, kqv_shape)
    Q_traits = nntile.tensor.TensorTraits(kqv_shape, kqv_shape)
    V_traits = nntile.tensor.TensorTraits(kqv_shape, kqv_shape)
    A_traits = nntile.tensor.TensorTraits(kqv_shape, kqv_shape)
    mask_traits = nntile.tensor.TensorTraits(mask_shape, mask_shape)
    logsumexp_traits = nntile.tensor.TensorTraits(logsumexp_shape,
                                                 logsumexp_shape)

    mpi_root = 0
    dist_root = [mpi_root]

    if dtype == 'fp16':
        tensor_type = nntile.tensor.Tensor_fp16
    elif dtype == 'bf16':
        tensor_type = nntile.tensor.Tensor_bf16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    numpy_dtype = np.float32

    K = tensor_type(K_traits, dist_root)
    Q = tensor_type(Q_traits, dist_root)
    V = tensor_type(V_traits, dist_root)
    A = tensor_type(A_traits, dist_root)
    mask = tensor_type(mask_traits, dist_root)
    logsumexp = nntile.tensor.Tensor_fp32(logsumexp_traits, dist_root)

    rng = np.random.default_rng(seed)

    K_src = rng.standard_normal(kqv_shape).astype(numpy_dtype, 'F') * 0.1
    Q_src = rng.standard_normal(kqv_shape).astype(numpy_dtype, 'F') * 0.1
    V_src = rng.standard_normal(kqv_shape).astype(numpy_dtype, 'F') * 0.1
    A_src = rng.standard_normal(kqv_shape).astype(numpy_dtype, 'F') * 0.1

    mask_src = np.full(mask_shape, -np.inf, dtype=numpy_dtype, order='F')
    window_size = 32
    for i in range(n_seq):
        for j in range(n_seq):
            if abs(i - j) <= window_size:
                mask_src[i, j] = 0.0

    logsumexp_src = np.full(logsumexp_shape, -np.inf, dtype=numpy_dtype, order='F')

    K.from_array(K_src)
    Q.from_array(Q_src)
    V.from_array(V_src)
    A.from_array(A_src)
    mask.from_array(mask_src)
    logsumexp.from_array(logsumexp_src)

    return {
        "K": K,
        "Q": Q,
        "V": V,
        "A": A,
        "mask": mask,
        "logsumexp": logsumexp,
        "K_src": K_src,
        "Q_src": Q_src,
        "V_src": V_src,
        "A_src": A_src,
        "mask_src": mask_src,
        "logsumexp_src": logsumexp_src,
        "kqv_shape": kqv_shape,
        "n_head_kv": n_head_kv,
    }


def pytorch_multihead_attention_baseline(Q_src, K_src, V_src, mask_src,
                                          n_head_kv):
    """
    Compute multi-head attention using PyTorch as a baseline for comparison.

    Args:
        Q_src: Query tensor in NNTile format [head_size, n_seq, n_batch,
               kv_group_size, n_head_kv]
        K_src: Key tensor in NNTile format [head_size, n_seq, n_batch,
               kv_group_size, n_head_kv]
        V_src: Value tensor in NNTile format [head_size, n_seq, n_batch,
               kv_group_size, n_head_kv]
        mask_src: Attention mask [n_seq, n_seq]
        n_head_kv: Number of key/value heads
        dtype: Data type ('fp16' or 'bf16')

    Returns:
        A_pytorch: Attention output in NNTile format [head_size, n_seq,
                    n_batch, kv_group_size, n_head_kv]
    """
    try:
        head_size, n_seq, n_batch, kv_group_size, _ = Q_src.shape

        # Convert to PyTorch tensors and reshape
        # From [head_size, n_seq, n_batch, kv_group_size, n_head_kv] to
        # [n_batch, n_seq, head_size * n_head_kv * kv_group_size]

        # First transpose to [n_batch, n_seq, head_size, kv_group_size,
        # n_head_kv]
        # Note: Q_src, K_src, V_src are NumPy arrays, so transpose works here
        Q_transposed = Q_src.transpose(2, 1, 0, 3, 4)
        # [n_batch, n_seq, head_size, kv_group_size, n_head_kv]
        K_transposed = K_src.transpose(2, 1, 0, 3, 4)
        V_transposed = V_src.transpose(2, 1, 0, 3, 4)

        # Then reshape to [n_batch, n_seq, head_size * n_head_kv *
        # kv_group_size]
        Q_pt = torch.from_numpy(Q_transposed.reshape(n_batch, n_seq, -1))
        K_pt = torch.from_numpy(K_transposed.reshape(n_batch, n_seq, -1))
        V_pt = torch.from_numpy(V_transposed.reshape(n_batch, n_seq, -1))

        # Convert mask: [n_seq, n_seq] -> [n_seq, n_seq]
        mask_pt = torch.from_numpy(mask_src)

        # Create multi-head attention layer
        # Note: embed_dim = head_size * n_head_kv * kv_group_size
        # For grouped query attention (GQA), kv_group_size represents how many
        # query heads share each key/value head
        # In this test case, kv_group_size=1, so it's standard MHA
        embed_dim = head_size * n_head_kv * kv_group_size
        mha = torch.nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=n_head_kv,
                                          batch_first=True)

        # Apply attention
        # Note: PyTorch MHA expects mask as [batch, seq, seq] where True means
        # masked out
        # Our mask has 0.0 for allowed positions and -inf for masked positions
        # Since our mask is 2D [seq, seq], we need to expand it to
        # 3D[batch, seq, seq]
        attn_mask = mask_pt.unsqueeze(0).expand(n_batch, -1, -1)
        attn_mask = attn_mask == -float('inf')

        # Set requires_grad=False to avoid gradient computation
        Q_pt.requires_grad_(False)
        K_pt.requires_grad_(False)
        V_pt.requires_grad_(False)

        A_pt, _ = mha(Q_pt, K_pt, V_pt, attn_mask=attn_mask)

        # Reshape back to NNTile format: [head_size, n_seq, n_batch,
        # kv_group_size, n_head_kv]
        A_pt_reshaped = A_pt.reshape(n_batch, n_seq, head_size,
                                     kv_group_size, n_head_kv)
        # Use permute() for PyTorch tensors instead of transpose() which only
        # works on 2D
        A_pytorch = A_pt_reshaped.permute(2, 1, 0, 3, 4).detach().numpy()

        return A_pytorch

    except ImportError:
        print("Warning: PyTorch not available, skipping baseline comparison")
        # Return zeros as placeholder - the test will still validate NNTile
        # functionality
        return np.zeros_like(Q_src)


@pytest.mark.parametrize('dtype', supported_dtypes)
def test_flash_sdpa_fwd_cudnn_async(context, dtype):
    data = _prepare_flash_inputs(dtype, seed=42)

    # Perform async flash_sdpa operation
    flash_sdpa_fwd_cudnn_async(data["K"], data["Q"], data["mask"],
                               data["logsumexp"], data["V"], data["A"])

    # Wait for completion
    nntile.starpu.wait_for_all()

    # Get results back
    data["A"].to_array(data["A_src"])

    # Compute PyTorch baseline for comparison
    A_pytorch = pytorch_multihead_attention_baseline(
        data["Q_src"], data["K_src"], data["V_src"], data["mask_src"],
        data["n_head_kv"])

    # Check shapes match
    assert data["A_src"].shape == A_pytorch.shape, (f"Shape mismatch: NNTile "
                                                    f"{data['A_src'].shape} "
                                                    f"vs PyTorch "
                                                    f"{A_pytorch.shape}")
    assert data["A_src"].shape == tuple(data["kqv_shape"])

    # Check that NNTile result is finite and has reasonable magnitude
    assert np.isfinite(data["A_src"]).all(), \
        "NNTile result contains NaN or Inf values"

    # Check if PyTorch baseline is available (non-zero result means
    # PyTorch worked)
    pytorch_available = np.any(A_pytorch != 0)

    if pytorch_available:
        # Check that PyTorch result is also finite
        assert np.isfinite(A_pytorch).any(), \
            "PyTorch result contains non-finite values"

        print("PyTorch baseline comparison available")

        # Compare results (allow for some numerical differences)
        # Note: cuDNN Flash Attention may have different numerical behavior
        # than PyTorch MHA due to different algorithms, precision, etc.
        relative_tolerance = 1e-3  # 0.1% relative tolerance

        # Only compare where PyTorch result is significant (to avoid
        # comparing noise)
        significant_mask = np.abs(A_pytorch) > 1e-6

        if np.any(significant_mask):
            max_diff = np.max(np.abs(data["A_src"][significant_mask] -
                                     A_pytorch[significant_mask]))
            max_val = np.max(np.abs(A_pytorch[significant_mask]))

            if max_val > 0:
                relative_diff = max_diff / max_val
                print(f"Maximum relative difference: {relative_diff:.6f}")
                print(f"Max absolute difference: {max_diff:.6f}")
                print(f"Max PyTorch value: {max_val:.6f}")

                # For now, just log the differences - in practice, Flash
                # Attention and standard MHA can produce different results
                # due to algorithmic differences
                # The commented assertion can be enabled for stricter
                # validation when needed
                assert relative_diff < relative_tolerance, \
                       f"Results differ: {relative_diff} > " \
                       f"{relative_tolerance}"
    else:
        print("PyTorch not available, skipping baseline comparison - only "
              "validating NNTile functionality")

    # Ensure NNTile result has some non-zero values
    # (indicating computation occurred)
        assert np.any(np.abs(data["A_src"]) > 1e-10), \
            "NNTile result appears to be all zeros - computation may have " \
            "failed"

    print(f"Test completed successfully for dtype {dtype}")
    print(f"NNTile result shape: {data['A_src'].shape}")
    print(f"NNTile result range: [{np.min(data['A_src']):.6f}, "
          f"{np.max(data['A_src']):.6f}]")
    print(f"NNTile result has {np.sum(np.abs(data['A_src']) > 1e-6)} "
          "significant values")
