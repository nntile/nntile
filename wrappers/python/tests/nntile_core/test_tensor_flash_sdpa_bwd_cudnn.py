# @file wrappers/python/tests/nntile_core/test_tensor_flash_sdpa_bwd_cudnn.py
# Tests for tensor::flash_sdpa_bwd_cudnn<T> Python wrapper

import math
import numpy as np
import pytest

import nntile
from nntile.functions import (
    flash_sdpa_fwd_cudnn_async,
    flash_sdpa_bwd_cudnn_async,
)

supported_dtypes = ["fp16", "bf16"]


def _prepare_flash_backward_inputs(dtype: str, seed: int = 17):
    head_size = 32
    n_seq = 32
    n_batch = 2
    kv_group_size = 1
    n_head_kv = 1

    kqv_shape = [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
    mask_shape = [n_seq, n_seq]
    logsumexp_shape = [n_seq, n_batch, kv_group_size, n_head_kv]

    K_traits = nntile.tensor.TensorTraits(kqv_shape, kqv_shape)
    Q_traits = nntile.tensor.TensorTraits(kqv_shape, kqv_shape)
    V_traits = nntile.tensor.TensorTraits(kqv_shape, kqv_shape)
    O_traits = nntile.tensor.TensorTraits(kqv_shape, kqv_shape)
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
    O = tensor_type(O_traits, dist)
    dO = tensor_type(O_traits, dist)
    dK = tensor_type(K_traits, dist)
    dQ = tensor_type(Q_traits, dist)
    dV = tensor_type(V_traits, dist)
    mask = tensor_type(mask_traits, dist)
    logsumexp = nntile.tensor.Tensor_fp32(logsumexp_traits, dist)

    rng = np.random.default_rng(seed)
    K_src = rng.standard_normal(kqv_shape).astype(np.float32, "F") * 0.1
    Q_src = rng.standard_normal(kqv_shape).astype(np.float32, "F") * 0.1
    V_src = rng.standard_normal(kqv_shape).astype(np.float32, "F") * 0.1
    dO_src = rng.standard_normal(kqv_shape).astype(np.float32, "F") * 0.05

    mask_src = np.full(mask_shape, -np.inf, dtype=np.float32, order="F")
    for i in range(n_seq):
        for j in range(n_seq):
            if abs(i - j) <= 8:
                mask_src[i, j] = 0.0

    logsumexp_src = np.zeros(
        logsumexp_shape, dtype=np.float32, order="F"
    )

    K.from_array(K_src)
    Q.from_array(Q_src)
    V.from_array(V_src)
    O.from_array(np.zeros_like(K_src))
    dO.from_array(dO_src)
    dK.from_array(np.zeros_like(K_src))
    dQ.from_array(np.zeros_like(K_src))
    dV.from_array(np.zeros_like(K_src))
    mask.from_array(mask_src)
    logsumexp.from_array(logsumexp_src)

    return {
        "K": K,
        "Q": Q,
        "V": V,
        "O": O,
        "dO": dO,
        "dK": dK,
        "dQ": dQ,
        "dV": dV,
        "mask": mask,
        "logsumexp": logsumexp,
        "K_src": K_src,
        "Q_src": Q_src,
        "V_src": V_src,
        "dO_src": dO_src,
        "mask_src": mask_src,
        "kqv_shape": kqv_shape,
        "head_size": head_size,
    }


def _flatten_to_batches(arr: np.ndarray) -> np.ndarray:
    head_size, n_seq, n_batch, kv_group_size, n_head_kv = arr.shape
    transposed = arr.transpose(2, 3, 4, 1, 0)
    return transposed.reshape(
        n_batch * kv_group_size * n_head_kv, n_seq, head_size
    )


def _unflatten_from_batches(flat: np.ndarray, shape):
    head_size, n_seq, n_batch, kv_group_size, n_head_kv = shape
    reshaped = flat.reshape(
        n_batch, kv_group_size, n_head_kv, n_seq, head_size
    )
    return reshaped.transpose(4, 3, 0, 1, 2)


def _torch_flash_sdpa_backward_reference(data):
    try:
        import torch
    except ImportError:
        return None

    q_t = torch.tensor(
        _flatten_to_batches(data["Q_src"]), dtype=torch.float32, requires_grad=True
    )
    k_t = torch.tensor(
        _flatten_to_batches(data["K_src"]), dtype=torch.float32, requires_grad=True
    )
    v_t = torch.tensor(
        _flatten_to_batches(data["V_src"]), dtype=torch.float32, requires_grad=True
    )
    mask = torch.tensor(data["mask_src"], dtype=torch.float32)
    dO = torch.tensor(
        _flatten_to_batches(data["dO_src"]), dtype=torch.float32
    )

    scale = 1.0 / math.sqrt(float(data["head_size"]))
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    scores = scores + mask.unsqueeze(0)
    attn = torch.softmax(scores, dim=-1)
    y = torch.matmul(attn, v_t)

    y.backward(dO)

    dQ = _unflatten_from_batches(q_t.grad.detach().cpu().numpy(), data["kqv_shape"])
    dK = _unflatten_from_batches(k_t.grad.detach().cpu().numpy(), data["kqv_shape"])
    dV = _unflatten_from_batches(v_t.grad.detach().cpu().numpy(), data["kqv_shape"])
    return {"dQ": dQ, "dK": dK, "dV": dV}


def _assert_close(actual: np.ndarray, reference: np.ndarray, tol=1e-2):
    diff = np.linalg.norm(actual - reference)
    ref_norm = np.linalg.norm(reference)
    limit = tol * (ref_norm if ref_norm != 0 else 1.0)
    assert diff <= limit + tol, f"diff {diff} exceeds limit {limit}"


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_flash_sdpa_bwd_cudnn_async(context, dtype):
    data = _prepare_flash_backward_inputs(dtype)

    flash_sdpa_fwd_cudnn_async(
        data["K"],
        data["Q"],
        data["mask"],
        data["logsumexp"],
        data["V"],
        data["O"],
    )
    nntile.starpu.wait_for_all()

    flash_sdpa_bwd_cudnn_async(
        data["K"],
        data["Q"],
        data["V"],
        data["O"],
        data["dO"],
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
    if ref is not None:
        _assert_close(dQ_out.astype(np.float32), ref["dQ"])
        _assert_close(dK_out.astype(np.float32), ref["dK"])
        _assert_close(dV_out.astype(np.float32), ref["dV"])
