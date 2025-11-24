# Unit tests for the SDPA layer mirror the general structure used by
# test_llama_attention: we generate deterministic inputs, run the layer in
# forward and forward+backward modes, and compare against a PyTorch baseline.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

import nntile
from nntile.layer import Sdpa
from nntile.tensor import (
    Tensor_bf16, Tensor_bool, Tensor_fp16, Tensor_fp32, TensorMoments,
    TensorTraits, clear_async)

dtype2tensor = {"fp32": Tensor_fp32, "fp16": Tensor_fp16, "bf16": Tensor_bf16}
dtype2tol = {
    "fp32": {"rtol": 1e-5, "atol": 1e-6},
    "fp16": {"rtol": 2e-3, "atol": 1e-4},
    "bf16": {"rtol": 1e-2, "atol": 5e-4},
}

nocuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no cuda"
)


@dataclass
class SDPATestParams:
    head_size: int
    n_seq: int
    n_batch: int
    kv_group_size: int
    n_head_kv: int
    n_seq_tile: int
    n_batch_tile: int
    kv_group_size_tile: int
    n_head_kv_tile: int
    seed: int = 0


single_tile = SDPATestParams(
    head_size=8,
    n_seq=6,
    n_seq_tile=6,
    n_batch=2,
    n_batch_tile=2,
    kv_group_size=1,
    kv_group_size_tile=1,
    n_head_kv=2,
    n_head_kv_tile=2,
)
multi_tile = SDPATestParams(
    head_size=8,
    n_seq=8,
    n_seq_tile=4,
    n_batch=3,
    n_batch_tile=1,
    kv_group_size=2,
    kv_group_size_tile=1,
    n_head_kv=4,
    n_head_kv_tile=2,
    seed=1,
)


@dataclass
class SDPAFlashTestParams(SDPATestParams):
    pass


flash_single_tile = SDPAFlashTestParams(
    head_size=64,
    n_seq=64,
    n_seq_tile=64,
    n_batch=2,
    n_batch_tile=2,
    kv_group_size=1,
    kv_group_size_tile=1,
    n_head_kv=2,
    n_head_kv_tile=2,
)

flash_multi_tile = SDPAFlashTestParams(
    head_size=64,
    n_seq=256,
    n_seq_tile=64,
    n_batch=3,
    n_batch_tile=1,
    kv_group_size=2,
    kv_group_size_tile=1,
    n_head_kv=4,
    n_head_kv_tile=2,
    seed=5,
)


def _make_tensor_moments(
    data: np.ndarray,
    tensor_type,
    basetile_shape: list[int],
    *,
    grad_required=True,
):
    arr = np.array(data, dtype=np.float32, order="F")
    traits = TensorTraits(list(arr.shape), basetile_shape)
    distr = [0] * traits.grid.nelems
    value = tensor_type(traits, distr)
    value.from_array(arr)
    grad = None
    if grad_required:
        grad = tensor_type(traits, distr)
        clear_async(grad)
    return TensorMoments(value, grad, grad_required)


def _make_mask(n_seq: int, n_seq_tile: int):
    # Some custom mask
    mask_np = np.tril(np.ones((n_seq, n_seq), dtype=bool), 32)
    traits = TensorTraits([n_seq, n_seq], [n_seq_tile, n_seq_tile])
    distr = [0] * traits.grid.nelems
    tensor = Tensor_bool(traits, distr)
    tensor.from_array(np.array(mask_np, dtype=bool, order="F"))
    return mask_np, tensor


def _tensor_to_numpy(tensor, dtype=np.float32):
    out = np.zeros(tensor.shape, dtype=dtype, order="F")
    tensor.to_array(out)
    return np.array(out, copy=True)


def _assert_tensor_close(actual: np.ndarray, reference: np.ndarray, tol):
    diff = np.linalg.norm(actual - reference)
    ref_norm = np.linalg.norm(reference)
    limit = tol["rtol"] * (ref_norm if ref_norm != 0 else 1.0) + tol["atol"]
    assert diff <= limit, f"diff {diff} exceeds limit {limit}"


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


def torch_sdpa_reference(q_np, k_np, v_np, mask_np, y_grad_np=None):
    requires_grad = y_grad_np is not None
    q_t = torch.tensor(q_np, dtype=torch.float32, requires_grad=requires_grad)
    k_t = torch.tensor(k_np, dtype=torch.float32, requires_grad=requires_grad)
    v_t = torch.tensor(v_np, dtype=torch.float32, requires_grad=requires_grad)
    scale = torch.tensor(1.0 / np.sqrt(q_np.shape[0]), dtype=torch.float32)
    scores = torch.einsum("hsbn,htbn->stbn", k_t, q_t) * scale
    mask_t = torch.tensor(mask_np, dtype=torch.bool).unsqueeze(-1) \
        .unsqueeze(-1)
    scores = torch.where(
        mask_t, scores, torch.full_like(scores, float("-inf")))
    attn = torch.softmax(scores, dim=0)
    out = torch.einsum("hsbn,stbn->htbn", v_t, attn)
    if not requires_grad:
        return out.detach().numpy(), None, None, None
    grad = torch.tensor(y_grad_np, dtype=torch.float32)
    out.backward(grad)
    return (
        out.detach().numpy(),
        q_t.grad.detach().numpy(),
        k_t.grad.detach().numpy(),
        v_t.grad.detach().numpy(),
    )


def generate_sdpa_inputs(
    dtype: str,
    params: SDPATestParams,
    *,
    require_backward: bool,
    mask_dtype: str,
) -> dict[str, np.ndarray | Sdpa]:
    rng = np.random.default_rng(params.seed)
    tensor_type = dtype2tensor[dtype]
    shape = (
        params.head_size,
        params.n_seq,
        params.n_batch,
        params.kv_group_size,
        params.n_head_kv,
    )
    basetile = [
        params.head_size,  # tile size must be equal to head_size
        params.n_seq_tile,
        params.n_batch_tile,
        params.kv_group_size_tile,
        params.n_head_kv_tile,
    ]

    q_np = rng.standard_normal(shape).astype(np.float32)
    k_np = rng.standard_normal(shape).astype(np.float32)
    v_np = rng.standard_normal(shape).astype(np.float32)

    q = _make_tensor_moments(
        q_np, tensor_type, basetile, grad_required=require_backward
    )
    k = _make_tensor_moments(
        k_np, tensor_type, basetile, grad_required=require_backward
    )
    v = _make_tensor_moments(
        v_np, tensor_type, basetile, grad_required=require_backward
    )

    mask_kind = mask_dtype.lower()
    if mask_kind == "bool":
        mask_np, mask_tensor = _make_mask(params.n_seq, params.n_seq_tile)
        flash_attention = False
    elif mask_kind == "float32":
        mask_traits = TensorTraits(
            [params.n_seq, params.n_seq],
            [params.n_seq_tile, params.n_seq_tile],
        )
        mask_tensor = tensor_type(mask_traits, [0] * mask_traits.grid.nelems)
        mask_values = np.tril(
            np.float32("-inf") * np.ones(
                (params.n_seq, params.n_seq),
                dtype=np.float32,
                order="F",
            ),
            -1
        )
        mask_tensor.from_array(mask_values)
        mask_np = mask_values == 0.0
        flash_attention = True
    elif mask_kind == "none":
        mask_tensor = None
        mask_np = np.ones(
            (params.n_seq, params.n_seq),
            dtype=bool,
            order="F",
        )
        flash_attention = True
    else:
        raise ValueError("mask_dtype must be 'bool', 'float32' or 'none'")

    layer = Sdpa.generate_simple(
        q=q,
        k=k,
        v=v,
        mask=mask_tensor,
        flash_attention=flash_attention,
        redux=False,
    )

    y_grad_np = None
    if require_backward:
        y_grad_np = rng.standard_normal(shape).astype(np.float32)
        layer.y.grad.from_array(
            np.array(y_grad_np, dtype=np.float32, order="F")
        )

    return {
        "layer": layer,
        "q_np": q_np,
        "k_np": k_np,
        "v_np": v_np,
        "mask_np": mask_np,
        "y_grad_np": y_grad_np,
    }


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("fp32", id="fp32"),
        pytest.param("fp16", id="fp16", marks=nocuda),
        pytest.param("bf16", id="bf16", marks=nocuda),
    ],
)
@pytest.mark.parametrize(
    "params",
    [
        pytest.param(single_tile, id="single_tile"),
        pytest.param(multi_tile, id="multi_tile")
    ],
)
class TestSDPAVanilla:
    def test_forward(self, context, dtype: str, params: SDPATestParams):
        inputs = generate_sdpa_inputs(
            dtype,
            params,
            require_backward=False,
            mask_dtype="bool",
        )
        layer = inputs["layer"]

        layer.forward_async()
        nntile.starpu.wait_for_all()

        q_ref = _flatten_sdpa_tensor(inputs["q_np"])
        k_ref = _flatten_sdpa_tensor(inputs["k_np"])
        v_ref = _flatten_sdpa_tensor(inputs["v_np"])
        y_nntile = _flatten_sdpa_tensor(_tensor_to_numpy(layer.y.value))
        y_ref, _, _, _ = torch_sdpa_reference(
            q_ref, k_ref, v_ref, inputs["mask_np"]
        )
        tol = dtype2tol[dtype]
        _assert_tensor_close(y_nntile, y_ref, tol)

    def test_forward_backward(
        self,
        context,
        dtype: str,
        params: SDPATestParams
    ):
        inputs = generate_sdpa_inputs(
            dtype,
            params,
            require_backward=True,
            mask_dtype="bool",
        )
        layer = inputs["layer"]

        layer.forward_async()
        nntile.starpu.wait_for_all()

        q_ref = _flatten_sdpa_tensor(inputs["q_np"])
        k_ref = _flatten_sdpa_tensor(inputs["k_np"])
        v_ref = _flatten_sdpa_tensor(inputs["v_np"])
        y_grad_ref = _flatten_sdpa_tensor(inputs["y_grad_np"])
        _, q_grad_ref, k_grad_ref, v_grad_ref = torch_sdpa_reference(
            q_ref,
            k_ref,
            v_ref,
            inputs["mask_np"],
            y_grad_ref,
        )

        layer.clear_gradients()
        layer.y.grad.from_array(
            np.array(inputs["y_grad_np"], dtype=np.float32, order="F")
        )

        layer.backward_async()
        nntile.starpu.wait_for_all()

        q_grad = _flatten_sdpa_tensor(_tensor_to_numpy(layer.q.grad))
        k_grad = _flatten_sdpa_tensor(_tensor_to_numpy(layer.k.grad))
        v_grad = _flatten_sdpa_tensor(_tensor_to_numpy(layer.v.grad))

        tol = dtype2tol[dtype]
        _assert_tensor_close(q_grad, q_grad_ref, tol)
        _assert_tensor_close(k_grad, k_grad_ref, tol)
        _assert_tensor_close(v_grad, v_grad_ref, tol)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "params",
    [
        pytest.param(flash_single_tile, id="flash_single_tile"),
        pytest.param(flash_multi_tile, id="flash_multi_tile"),
    ],
)
@nocuda
class TestSDPAFlash:
    def test_forward(
        self,
        context,
        dtype: str,
        params: SDPAFlashTestParams,
    ):
        inputs = generate_sdpa_inputs(
            dtype,
            params,
            require_backward=False,
            mask_dtype="float32",
        )
        layer = inputs["layer"]
        assert tuple(layer.flash_logsumexp.shape) == tuple(
            layer.q.value.shape[1:]
        )

        layer.forward_async()
        nntile.starpu.wait_for_all()

        q_ref = _flatten_sdpa_tensor(inputs["q_np"])
        k_ref = _flatten_sdpa_tensor(inputs["k_np"])
        v_ref = _flatten_sdpa_tensor(inputs["v_np"])
        y_ref, _, _, _ = torch_sdpa_reference(
            q_ref, k_ref, v_ref, inputs["mask_np"]
        )

        y_nntile = _flatten_sdpa_tensor(_tensor_to_numpy(layer.y.value))
        tol = dtype2tol[dtype]
        _assert_tensor_close(y_nntile, y_ref, tol)

    def test_backward(
        self,
        context,
        dtype: str,
        params: SDPAFlashTestParams,
    ):
        inputs = generate_sdpa_inputs(
            dtype,
            params,
            require_backward=True,
            mask_dtype="float32",
        )
        layer = inputs["layer"]

        layer.forward_async()
        nntile.starpu.wait_for_all()

        q_ref = _flatten_sdpa_tensor(inputs["q_np"])
        k_ref = _flatten_sdpa_tensor(inputs["k_np"])
        v_ref = _flatten_sdpa_tensor(inputs["v_np"])
        y_grad_ref = _flatten_sdpa_tensor(inputs["y_grad_np"])
        _, q_grad_ref, k_grad_ref, v_grad_ref = torch_sdpa_reference(
            q_ref,
            k_ref,
            v_ref,
            inputs["mask_np"],
            y_grad_ref,
        )

        layer.clear_gradients()
        layer.y.grad.from_array(
            np.array(inputs["y_grad_np"], dtype=np.float32, order="F")
        )

        layer.backward_async()
        nntile.starpu.wait_for_all()

        q_grad = _flatten_sdpa_tensor(_tensor_to_numpy(layer.q.grad))
        k_grad = _flatten_sdpa_tensor(_tensor_to_numpy(layer.k.grad))
        v_grad = _flatten_sdpa_tensor(_tensor_to_numpy(layer.v.grad))

        tol = dtype2tol[dtype]
        _assert_tensor_close(q_grad, q_grad_ref, tol)
        _assert_tensor_close(k_grad, k_grad_ref, tol)
        _assert_tensor_close(v_grad, v_grad_ref, tol)


def test_flash_logsumexp_shape_validation(context):
    inputs = generate_sdpa_inputs(
        dtype="fp16",
        params=flash_single_tile,
        require_backward=False,
        mask_dtype="float32",
    )
    layer = inputs["layer"]
    wrong_shape = list(layer.flash_logsumexp.shape)
    wrong_shape = wrong_shape[1:] + wrong_shape[:1]
    traits = TensorTraits(wrong_shape, wrong_shape)
    wrong_logsumexp = Tensor_fp32(traits, [0] * traits.grid.nelems)

    assert tuple(wrong_logsumexp.shape) != tuple(layer.flash_logsumexp.shape)

    with pytest.raises(ValueError, match="flash_logsumexp"):
        Sdpa(
            q=layer.q,
            k=layer.k,
            v=layer.v,
            y=layer.y,
            mask=layer.mask,
            flash_attention=True,
            flash_logsumexp=wrong_logsumexp,
            redux=False,
        )
