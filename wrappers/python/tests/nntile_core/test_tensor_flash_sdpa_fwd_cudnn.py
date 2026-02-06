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
# This test validates the cuDNN Flash Attention implementation by running the
# NNTile flash_sdpa_fwd_cudnn_async function and comparing its output against a
# float32 PyTorch reference implementation of scaled dot-product attention.
#
# @version 1.1.0

import numpy as np
import pytest
import torch

import nntile
from nntile.functions import flash_sdpa_fwd_cudnn_async

# Only test BF16 and FP16 as per cuDNN limitations
supported_dtypes = ["fp16", "bf16"]
dtype2tol = {
    "fp16": {"rtol": 2e-3, "atol": 1e-4},
    "bf16": {"rtol": 1e-2, "atol": 5e-4},
}

nocuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no cuda"
)


def _prepare_flash_inputs(dtype, seed=42):
    head_size = 32
    n_seq = 128
    n_batch = 2
    kv_group_size = 2
    n_head_kv = 2
    n_seq_tile = 64
    n_batch_tile = 1
    kv_group_size_tile = 1
    n_head_kv_tile = 1

    kqv_shape = [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
    kqv_basetile = [
        head_size,
        n_seq_tile,
        n_batch_tile,
        kv_group_size_tile,
        n_head_kv_tile,
    ]
    mask_shape = [n_seq, n_seq]
    mask_basetile = [n_seq_tile, n_seq_tile]
    logsumexp_shape = [n_seq, n_batch, kv_group_size, n_head_kv]
    logsumexp_basetile = [
        n_seq_tile,
        n_batch_tile,
        kv_group_size_tile,
        n_head_kv_tile,
    ]

    K_traits = nntile.tensor.TensorTraits(kqv_shape, kqv_basetile)
    Q_traits = nntile.tensor.TensorTraits(kqv_shape, kqv_basetile)
    V_traits = nntile.tensor.TensorTraits(kqv_shape, kqv_basetile)
    A_traits = nntile.tensor.TensorTraits(kqv_shape, kqv_basetile)
    mask_traits = nntile.tensor.TensorTraits(mask_shape, mask_basetile)
    logsumexp_traits = nntile.tensor.TensorTraits(
        logsumexp_shape, logsumexp_basetile
    )

    if dtype == "fp16":
        tensor_type = nntile.tensor.Tensor_fp16
    elif dtype == "bf16":
        tensor_type = nntile.tensor.Tensor_bf16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    K = tensor_type(K_traits)
    Q = tensor_type(Q_traits)
    V = tensor_type(V_traits)
    A = tensor_type(A_traits)
    mask = tensor_type(mask_traits)
    logsumexp = nntile.tensor.Tensor_fp32(logsumexp_traits)

    rng = np.random.default_rng(seed)

    K_src = rng.standard_normal(kqv_shape).astype(np.float32, "F") * 0.1
    Q_src = rng.standard_normal(kqv_shape).astype(np.float32, "F") * 0.1
    V_src = rng.standard_normal(kqv_shape).astype(np.float32, "F") * 0.1
    A_src = np.zeros(kqv_shape, dtype=np.float32, order="F")

    mask_src = np.full(
        mask_shape, -np.inf, dtype=np.float32, order="F"
    )
    # Causal mask works, but some other mask may fail for some strange reason
    # Seems like cuDNN does not handle properly certain situations
    for i in range(n_seq):
        for j in range(n_seq):
            if abs(i - j) <= 32:
                mask_src[i, j] = 0.0

    logsumexp_src = np.full(
        logsumexp_shape, -np.inf, dtype=np.float32, order="F"
    )

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
        "kqv_shape": kqv_shape,
    }


def _flatten_sdpa_tensor(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim != 5:
        raise ValueError("SDPA tensors must have 4 or 5 dimensions")
    head_size, n_seq, n_batch, kv_group_size, n_head_kv = tensor.shape
    return np.reshape(
        tensor,
        (head_size, n_seq, n_batch, kv_group_size * n_head_kv),
        order="F",
    )


def flash_attention_reference(Q_src, K_src, V_src, mask_src):
    q_flat = _flatten_sdpa_tensor(Q_src).astype(np.float32, order="F")
    k_flat = _flatten_sdpa_tensor(K_src).astype(np.float32, order="F")
    v_flat = _flatten_sdpa_tensor(V_src).astype(np.float32, order="F")

    head_size = q_flat.shape[0]
    q_t = torch.tensor(q_flat, dtype=torch.float32)
    k_t = torch.tensor(k_flat, dtype=torch.float32)
    v_t = torch.tensor(v_flat, dtype=torch.float32)

    scale = torch.tensor(1.0 / np.sqrt(head_size), dtype=torch.float32)
    scores = torch.einsum("hsbn,htbn->stbn", k_t, q_t) * scale
    scores = scores + torch.tensor(mask_src, dtype=torch.float32) \
        .unsqueeze(-1).unsqueeze(-1)
    attn = torch.softmax(scores, dim=0)
    logsumexp = torch.logsumexp(scores, dim=0)
    out = torch.einsum("hsbn,stbn->htbn", v_t, attn).detach().cpu().numpy()

    out_np = np.reshape(out, Q_src.shape, order="F")
    lse_np = np.reshape(
        logsumexp.detach().cpu().numpy(),
        (Q_src.shape[1], Q_src.shape[2], Q_src.shape[3], Q_src.shape[4]),
        order="F",
    )
    return out_np, lse_np


def _assert_tensor_relative_close(actual: np.ndarray, reference: np.ndarray,
                                  tol):
    diff = np.linalg.norm(actual - reference)
    ref_norm = np.linalg.norm(reference)
    limit = tol["rtol"] * (ref_norm if ref_norm != 0 else 1.0) + tol["atol"]
    assert diff <= limit, f"relative diff {diff} exceeds limit {limit}"


@nocuda
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_flash_sdpa_fwd_cudnn_async(context, dtype):
    pytest.xfail("under development")

    data = _prepare_flash_inputs(dtype, seed=42)

    flash_sdpa_fwd_cudnn_async(
        data["K"], data["Q"], data["mask"], data["logsumexp"],
        data["V"], data["A"]
    )
    nntile.starpu.wait_for_all()

    a_out = np.empty_like(data["A_src"], order="F")
    data["A"].to_array(a_out)

    a_ref, lse_ref = flash_attention_reference(
        data["Q_src"], data["K_src"], data["V_src"], data["mask_src"]
    )

    lse_out = np.empty(data["logsumexp"].shape, dtype=np.float32, order="F")
    data["logsumexp"].to_array(lse_out)

    assert a_out.shape == a_ref.shape == tuple(data["kqv_shape"])
    assert lse_out.shape == lse_ref.shape
    assert np.isfinite(a_out).all()
    assert np.isfinite(a_ref).all()
    assert np.isfinite(lse_ref).all()
    assert np.isfinite(lse_out).all()

    tol = dtype2tol[dtype]
    _assert_tensor_relative_close(lse_out, lse_ref, tol)
    _assert_tensor_relative_close(a_out, a_ref, tol)
