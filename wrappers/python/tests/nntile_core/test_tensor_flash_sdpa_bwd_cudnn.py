# @file wrappers/python/tests/nntile_core/test_tensor_flash_sdpa_bwd_cudnn.py
# Tests for tensor::flash_sdpa_bwd_cudnn<T> Python wrapper
#
# This test runs the cuDNN flash backward kernel and checks dQ/dK/dV against a
# float32 PyTorch reference of scaled dot-product attention.

import math

import numpy as np
import pytest
import torch

import nntile
from nntile.functions import (
    clear_async, flash_sdpa_bwd_cudnn_async, flash_sdpa_fwd_cudnn_async)

supported_dtypes = ["fp16", "bf16"]
dtype2tol = {
    "fp16": {"rtol": 2e-3, "atol": 1e-4},
    "bf16": {"rtol": 1e-2, "atol": 5e-4},
}

nocuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no cuda"
)


def _prepare_flash_backward_inputs(dtype: str, seed: int = 17):
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
    logsumexp_traits = nntile.tensor.TensorTraits(
        logsumexp_shape, logsumexp_shape
    )

    mpi_root = 0
    dist = [mpi_root]

    if dtype == "fp16":
        tensor_type = nntile.tensor.Tensor_fp16
    elif dtype == "bf16":
        tensor_type = nntile.tensor.Tensor_bf16
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    K = tensor_type(K_traits, dist)
    Q = tensor_type(Q_traits, dist)
    V = tensor_type(V_traits, dist)
    A = tensor_type(A_traits, dist)
    dA = tensor_type(A_traits, dist)
    dK = tensor_type(K_traits, dist)
    dQ = tensor_type(Q_traits, dist)
    dV = tensor_type(V_traits, dist)
    mask = tensor_type(mask_traits, dist)
    logsumexp = nntile.tensor.Tensor_fp32(logsumexp_traits, dist)

    rng = np.random.default_rng(seed)
    K_src = rng.standard_normal(kqv_shape).astype(np.float32, "F") * 0.1
    Q_src = rng.standard_normal(kqv_shape).astype(np.float32, "F") * 0.1
    V_src = rng.standard_normal(kqv_shape).astype(np.float32, "F") * 0.1
    dA_src = rng.standard_normal(kqv_shape).astype(np.float32, "F") * 0.05

    mask_src = np.full(mask_shape, -np.inf, dtype=np.float32, order="F")
    # Causal mask works, but some other mask may fail for some strange reason
    # Seems like cuDNN does not handle properly certain situations
    for i in range(n_seq):
        for j in range(n_seq):
            if i <= j:
                mask_src[i, j] = 0.0

    logsumexp_src = np.full(
        logsumexp_shape, -np.inf, dtype=np.float32, order="F"
    )

    K.from_array(K_src)
    Q.from_array(Q_src)
    V.from_array(V_src)
    A.from_array(np.zeros_like(K_src))
    dA.from_array(dA_src)
    clear_async(dK)
    clear_async(dQ)
    clear_async(dV)
    mask.from_array(mask_src)
    logsumexp.from_array(logsumexp_src)

    return {
        "K": K,
        "Q": Q,
        "V": V,
        "A": A,
        "dA": dA,
        "dK": dK,
        "dQ": dQ,
        "dV": dV,
        "mask": mask,
        "logsumexp": logsumexp,
        "K_src": K_src,
        "Q_src": Q_src,
        "V_src": V_src,
        "dA_src": dA_src,
        "mask_src": mask_src,
        "kqv_shape": kqv_shape,
        "head_size": head_size,
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


def _torch_flash_sdpa_backward_reference(data):
    q_flat = _flatten_sdpa_tensor(data["Q_src"]).astype(np.float32, order="F")
    k_flat = _flatten_sdpa_tensor(data["K_src"]).astype(np.float32, order="F")
    v_flat = _flatten_sdpa_tensor(data["V_src"]).astype(np.float32, order="F")
    da_flat = _flatten_sdpa_tensor(data["dA_src"]).astype(np.float32, "F")

    q_t = torch.tensor(q_flat, dtype=torch.float32, requires_grad=True)
    k_t = torch.tensor(k_flat, dtype=torch.float32, requires_grad=True)
    v_t = torch.tensor(v_flat, dtype=torch.float32, requires_grad=True)
    da_t = torch.tensor(da_flat, dtype=torch.float32)

    scale = torch.tensor(
        1.0 / math.sqrt(float(data["head_size"])), dtype=torch.float32
    )
    scores = torch.einsum("hsbn,htbn->stbn", k_t, q_t) * scale
    scores = scores + torch.tensor(data["mask_src"], dtype=torch.float32) \
        .unsqueeze(-1).unsqueeze(-1)
    attn = torch.softmax(scores, dim=0)
    y = torch.einsum("hsbn,stbn->htbn", v_t, attn)

    y.backward(da_t)

    dQ = np.reshape(
        q_t.grad.detach().cpu().numpy(), data["kqv_shape"], order="F"
    )
    dK = np.reshape(
        k_t.grad.detach().cpu().numpy(), data["kqv_shape"], order="F"
    )
    dV = np.reshape(
        v_t.grad.detach().cpu().numpy(), data["kqv_shape"], order="F"
    )
    return {"dQ": dQ, "dK": dK, "dV": dV}


@nocuda
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_flash_sdpa_bwd_cudnn_async(context, dtype):
    data = _prepare_flash_backward_inputs(dtype)

    flash_sdpa_fwd_cudnn_async(
        data["K"],
        data["Q"],
        data["mask"],
        data["logsumexp"],
        data["V"],
        data["A"],
    )
    nntile.starpu.wait_for_all()

    flash_sdpa_bwd_cudnn_async(
        data["K"],
        data["Q"],
        data["V"],
        data["A"],
        data["dA"],
        data["mask"],
        data["logsumexp"],
        data["dK"],
        data["dQ"],
        data["dV"],
    )
    nntile.starpu.wait_for_all()

    dQ_out = np.zeros_like(data["Q_src"])
    dK_out = np.zeros_like(data["K_src"])
    dV_out = np.zeros_like(data["V_src"])
    data["dQ"].to_array(dQ_out)
    data["dK"].to_array(dK_out)
    data["dV"].to_array(dV_out)

    assert np.isfinite(dQ_out).all()
    assert np.isfinite(dK_out).all()
    assert np.isfinite(dV_out).all()
    assert np.any(np.abs(dQ_out) > 1e-7)
    assert np.any(np.abs(dK_out) > 1e-7)
    assert np.any(np.abs(dV_out) > 1e-7)

    ref = _torch_flash_sdpa_backward_reference(data)
    tol = dtype2tol[dtype]
    np.testing.assert_allclose(dQ_out, ref["dQ"], rtol=tol["rtol"],
                               atol=tol["atol"])
    np.testing.assert_allclose(dK_out, ref["dK"], rtol=tol["rtol"],
                               atol=tol["atol"])
    np.testing.assert_allclose(dV_out, ref["dV"], rtol=tol["rtol"],
                               atol=tol["atol"])
