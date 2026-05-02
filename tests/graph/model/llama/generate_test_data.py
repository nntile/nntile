#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file tests/graph/model/llama/generate_test_data.py
# Generate Llama building-block test data in safetensors format.
# Each invocation produces data for a single block (--block).
#
# @version 1.1.0

"""Generate reference test data for NNTile Llama C++ tests.

For each block the script creates a ``llama_<block>.safetensors`` file
that stores NNTile-layout weights, input tensor(s), reference forward output,
and backward reference gradients. The ``mlp`` / ``decoder`` / ``decoder_gqa``
/ ``model`` / ``model_gqa`` / ``causal`` / ``causal_gqa`` blocks also write
paired ``.json`` sidecars (geometry, tolerances) read by the corresponding C++
graph tests.

Uses **HuggingFace Transformers** (``modeling_llama``) plus NumPy layout
wrangling only — no NNTile Python runtime or StarPU.

``mlp`` uses ``MHA_DIMS`` (hidden=8). ``decoder`` / ``decoder_gqa`` /
``model`` / ``model_gqa`` / ``causal`` / ``causal_gqa`` use
``DECODER_MHA_DIMS`` / ``DECODER_GQA_DIMS`` (aliases of the graph attention
reference bundles). ``attention`` / ``attention_gqa`` use
``ATTENTION_MHA_DIMS`` / ``ATTENTION_GQA_DIMS`` (``test_llama_attention``
``single_tile`` + C++ graph attention tests).

Attention ``q_weight`` / ``k_weight`` / ``v_weight`` / ``o_weight`` are the
same numeric values as HuggingFace ``Linear`` weights, reshaped to the 3D/4D
layouts expected by the graph module, then passed through :func:`fortran_order`
so byte layout matches NNTile Fortran tiles. PyTorch runs forward and backward
with the **original** HF weights (no in-place Q/K rewrite).

For ``attention`` / ``attention_gqa`` blocks, Q/K weights use the same
RoPE head-dim interleaving as the Python layer (Q: ``rotate_tensor_in`` on the
head_size axis; GQA ``o_weight`` uses the ``from_torch`` reshape plus
``moveaxis(1, 2)``). ``rope_sin`` / ``rope_cos`` are
the first half-channels of ``LlamaRotaryEmbedding`` cos/sin, reshaped to
``(head_dim/2, seq, batch)`` and passed through :func:`fortran_order` so the
byte layout matches the C++ graph ``bind_data`` convention. RoPE is built like
``test_llama_attention.generate_inputs``: first argument ``v_proj.weight``,
``position_ids`` in ``(batch, seq)`` from a NumPy RNG.Forward and backward use
``LlamaAttention`` with ``_attn_implementation="eager"`` and the same
``(cos, sin)`` tensors as in the Python test.

Optional causal self-attention matches ``test_llama_attention``: additive
``attention_mask`` from the upper-triangular bool pattern; the graph tests
load ``attn_mask`` as float32 ``(seq, seq)`` in Fortran layout (1 = keep
logits), converted to BOOL in C++ for ``sdpa_eager`` masking.

Extra MHA/GQA safetensors (identity RoPE / causal / both) are written by
``--write-attention-rope-mask-variants`` (CTest fixture for
``llama_attention``).
Identity RoPE still stores ``rope_cos`` / ``rope_sin`` so the C++ graph matches
the PyTorch path (no null RoPE tensors).

Each graph attention safetensors bundle has a **paired** JSON sidecar with the
same basename (e.g. ``llama_attention.json`` next to
``llama_attention.safetensors``): ``Llama`` attention geometry, tensor
layout (``sequence_length``, ``batch``), and forward/backward tolerances.
C++ tests load the JSON to build ``LlamaConfig``, construct ``LlamaAttention``,
then ``load()`` the sibling ``.safetensors``.

The ``mlp`` block likewise writes ``llama_mlp.json`` next to
``llama_mlp.safetensors`` (``hidden_size``, ``intermediate_size``, head
counts, ``sequence_length``, ``batch``, tolerances) for ``test_llama_mlp``.

The ``decoder`` / ``decoder_gqa`` bundles add ``rope_cos`` / ``rope_sin`` in
the same layout as the attention tests (first half-channels, Fortran layout),
plus ``llama_decoder.json`` / ``llama_decoder_gqa.json`` for
``test_llama_decoder``.

The ``model`` / ``model_gqa`` bundles add the same RoPE tensors (from
``LlamaModel.rotary_emb`` on token embeddings, matching HF), ``attn_mask`` for
causal ``sdpa_eager`` (as in the attention tests), and ``llama_model.json`` /
``llama_model_gqa.json`` for ``test_llama_model``.

The ``causal`` / ``causal_gqa`` bundles do the same for ``LlamaForCausalLM``
(logits reference) plus ``llama_causal.json`` / ``llama_causal_gqa.json`` for
``test_llama_causal`` (JSON schema matches ``model`` fixtures).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention as PtAttention, LlamaDecoderLayer as PtDecoderLayer,
    LlamaForCausalLM as PtCausalLM, LlamaMLP as PtMLP, LlamaModel as PtModel,
    LlamaRotaryEmbedding)

# ── Test dimension bundles ────────────────────────────────────────────────


@dataclass
class TestDims:
    hidden: int
    intermediate: int
    n_heads: int
    kv_heads: int
    seq: int
    batch: int
    vocab: int
    num_layers: int
    rms_eps: float = 1e-6

    @property
    def head_size(self) -> int:
        return self.hidden // self.n_heads

    @property
    def use_gqa(self) -> bool:
        return self.kv_heads < self.n_heads

    @property
    def kv_group_size(self) -> int:
        return self.n_heads // self.kv_heads


# Small bundles for ``mlp`` / ``causal`` and for ``causal_gqa`` (keeps these
# safetensors light).
MHA_DIMS = TestDims(
    hidden=8, intermediate=16, n_heads=1, kv_heads=1,
    seq=4, batch=2, vocab=100, num_layers=2,
)

GQA_DIMS = TestDims(
    hidden=8, intermediate=16, n_heads=4, kv_heads=2,
    seq=4, batch=2, vocab=100, num_layers=2,
)

# ``wrappers/python/tests/model/test_llama_attention.py`` (``single_tile``):
# head_size=64, n_head=8, n_head_kv=4, seq=64, batch=3. True MHA uses one head
# (hidden=64) for the ``attention`` block; GQA uses 8/4 and hidden=512.
ATTENTION_MHA_DIMS = TestDims(
    hidden=64, intermediate=256, n_heads=1, kv_heads=1,
    seq=64, batch=3, vocab=100, num_layers=1,
)
ATTENTION_GQA_DIMS = TestDims(
    hidden=512, intermediate=2048, n_heads=8, kv_heads=4,
    seq=64, batch=3, vocab=100, num_layers=1,
)

# Graph ``test_llama_decoder`` / ``test_llama_model`` use the same geometry as
# the attention bundles.
DECODER_MHA_DIMS = ATTENTION_MHA_DIMS
DECODER_GQA_DIMS = ATTENTION_GQA_DIMS
MODEL_MHA_DIMS = DECODER_MHA_DIMS
MODEL_GQA_DIMS = DECODER_GQA_DIMS
CAUSAL_MHA_DIMS = MODEL_MHA_DIMS
CAUSAL_GQA_DIMS = MODEL_GQA_DIMS

# ── Helpers ──────────────────────────────────────────────────────────────


def fortran_order(arr: np.ndarray) -> np.ndarray:
    """Return C-contiguous array matching NNTile column-major layout.

    ``safetensors`` stores C-order (row-major) flat bytes via ``tobytes()``.
    NNTile reads those bytes linearly into Fortran-order (column-major) tiles.
    Ravelling in F-order and reshaping in C-order makes the C-order flat
    representation equal the column-major element sequence that NNTile expects.
    """
    a = np.asarray(arr, dtype=np.float32)
    return a.ravel("F").reshape(a.shape)


def fortran_order_int64(arr: np.ndarray) -> np.ndarray:
    """Same layout remap as :func:`fortran_order` for int64 position indices."""
    a = np.asarray(arr, dtype=np.int64)
    return a.ravel("F").reshape(a.shape)


def _make_config(dims: TestDims) -> LlamaConfig:
    return LlamaConfig(
        hidden_size=dims.hidden,
        intermediate_size=dims.intermediate,
        num_attention_heads=dims.n_heads,
        num_key_value_heads=dims.kv_heads,
        num_hidden_layers=dims.num_layers,
        vocab_size=dims.vocab,
        rms_norm_eps=dims.rms_eps,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        _attn_implementation="eager",
        base_model_tp_plan=None,
        # Parity with ``wrappers/python/tests/model/test_llama_attention.py``
        attention_bias=False,
        pretraining_tp=1,
    )


# ── Weight-extraction (PyTorch → NNTile layout) ─────────────────────────


def _linear(linear: torch.nn.Linear) -> np.ndarray:
    """PT Linear weight ``(out, in)`` C → NNTile ``(in, out)`` Fortran."""
    return fortran_order(linear.weight.detach().numpy().T)


def _attention_weight_arrays(
    attn: PtAttention, dims: TestDims,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """HF ``(out,in)`` Q/K/V/O reshaped to graph ``LlamaAttention``
    weight layouts.

    No RoPE-related half-dim interleaving; values match ``q_proj`` /
    ``k_proj`` / ``v_proj`` / ``o_proj`` as initialized by HuggingFace.

    Returns ``(q_arr, k_arr, v, o, o_arr)`` — ``v`` / ``o`` are raw torch
    arrays; ``o_arr`` is the NNTile-shaped ``o`` view used for ``o_weight``.
    """
    q = attn.q_proj.weight.detach().numpy()
    k = attn.k_proj.weight.detach().numpy()
    v = attn.v_proj.weight.detach().numpy()
    o = attn.o_proj.weight.detach().numpy()
    n_emb = q.shape[1]

    if dims.use_gqa:
        q_arr = q.reshape(
            dims.kv_heads, dims.kv_group_size, dims.head_size, n_emb,
        ).transpose(1, 0, 2, 3)
        # Match ``from_torch``: reshape to
        # ``(n_emb, n_head_kv, kv_group, head)``
        # then ``moveaxis(1, 2)`` → NNTile
        # ``(n_emb, kv_group, n_head_kv, head)``.
        o_tmp = o.reshape(
            n_emb, dims.kv_heads, dims.kv_group_size, dims.head_size,
        )
        o_arr = np.moveaxis(o_tmp, 1, 2)
    else:
        q_arr = q.reshape(dims.n_heads, dims.head_size, n_emb)
        o_arr = o.reshape(n_emb, dims.n_heads, dims.head_size)

    k_arr = k.reshape(dims.kv_heads, dims.head_size, n_emb)
    q_arr = np.asarray(q_arr, dtype=np.float32).copy()
    k_arr = np.asarray(k_arr, dtype=np.float32).copy()
    return q_arr, k_arr, v, o, o_arr


def _attn(attn: PtAttention, prefix: str, dims: TestDims) -> \
    dict[str, np.ndarray]:
    q_arr, k_arr, v, _o, o_arr = _attention_weight_arrays(attn, dims)

    def rotate_tensor_in(x: np.ndarray, axis: int) -> np.ndarray:
        if axis == 0:
            new_shape = (1, x.shape[0], np.prod(x.shape[1:]))
        elif axis == x.ndim - 1:
            new_shape = (np.prod(x.shape[:-1]), x.shape[-1], 1)
        else:
            new_shape = (
                np.prod(x.shape[:axis]),
                x.shape[axis],
                np.prod(x.shape[axis + 1 :]),
            )
        x_reshaped = x.reshape(new_shape)
        mid = x.shape[axis] // 2
        y_reshaped = np.empty_like(x_reshaped)
        y_reshaped[:, 0::2, :] = x_reshaped[:, :mid, :]
        y_reshaped[:, 1::2, :] = x_reshaped[:, mid:, :]
        return y_reshaped.reshape(x.shape)

    n_emb = int(attn.q_proj.weight.shape[1])
    # Q RoPE layout: rotate along head_size (axis 1 for 3D, axis 2 for 4D GQA),
    # matching ``LlamaAttention_nntile.from_torch`` (rotate_tensor_in(..., 2)).
    q_rot_axis = q_arr.ndim - 2
    return {
        f"{prefix}.q_weight": fortran_order(
            rotate_tensor_in(q_arr, q_rot_axis),
        ),
        f"{prefix}.k_weight": fortran_order(rotate_tensor_in(k_arr, 1)),
        f"{prefix}.v_weight": fortran_order(
            v.reshape(dims.kv_heads, dims.head_size, n_emb)
        ),
        f"{prefix}.o_weight": fortran_order(o_arr),
    }


def _rms(norm, prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}.gamma": fortran_order(norm.weight.detach().numpy())}


def _embed(embed, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.vocab": fortran_order(embed.weight.detach().numpy().T)
    }


def _mlp(mlp, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.gate_proj.weight": _linear(mlp.gate_proj),
        f"{prefix}.up_proj.weight": _linear(mlp.up_proj),
        f"{prefix}.down_proj.weight": _linear(mlp.down_proj),
    }


def _decoder_layer(
    layer, prefix: str, dims: TestDims) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {}
    d.update(_rms(layer.input_layernorm, f"{prefix}.input_norm"))
    d.update(_attn(layer.self_attn, f"{prefix}.attention", dims))
    d.update(
        _rms(layer.post_attention_layernorm, f"{prefix}.post_attn_norm")
    )
    d.update(_mlp(layer.mlp, f"{prefix}.mlp"))
    return d


def _model_weights(
    model, prefix: str, dims: TestDims) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {}
    d.update(_embed(model.embed_tokens, f"{prefix}.embed_tokens"))
    d.update(_rms(model.norm, f"{prefix}.norm"))
    for i, layer in enumerate(model.layers):
        d.update(_decoder_layer(layer, f"{prefix}.layers_{i}", dims))
    return d


# ── Input / output helpers ───────────────────────────────────────────────


def _hidden_input(rng, dims: TestDims, scale: float = 0.1):
    """Random hidden-state input: NNTile (H,S,B) Fortran + PT (B,S,H)."""
    x = rng.standard_normal(
        (dims.hidden, dims.seq, dims.batch)
    ).astype(np.float32) * scale
    x_nt = fortran_order(x)
    x_pt = torch.tensor(x.transpose(2, 1, 0).copy(), requires_grad=True)
    return x_nt, x_pt


def _grad_output(rng, pt_out: torch.Tensor, scale: float = 0.1):
    """Random upstream gradient matching output shape (B,S,D).

    Returns ``(grad_nt, grad_pt)`` — NNTile ``(D,S,B)`` Fortran and PyTorch
    ``(B,S,D)`` tensors.
    """
    g = (rng.standard_normal(pt_out.shape).astype(np.float32) * scale)
    g_pt = torch.tensor(g)
    g_nt = fortran_order(g.transpose(2, 1, 0))
    return g_nt, g_pt


def _ids_input(rng, dims: TestDims):
    """Random token-id input: NNTile ``(S,B)`` Fortran + PT ``(B,S)``."""
    ids = rng.integers(
        0, dims.vocab, size=(dims.seq, dims.batch)).astype(np.int64)
    ids_nt = ids.ravel("F").reshape(ids.shape)
    ids_pt = torch.tensor(ids.T.copy(), dtype=torch.long)
    return ids_nt, ids_pt


def _out_to_nntile(pt_out: torch.Tensor) -> np.ndarray:
    """PT output ``(B, S, D)`` → NNTile ``(D, S, B)`` Fortran."""
    return fortran_order(pt_out.detach().numpy().transpose(2, 1, 0))


def _rope_half_from_hf(
    cos: torch.Tensor, sin: torch.Tensor, dims: TestDims,
) -> tuple[np.ndarray, np.ndarray]:
    """HF ``(B,S,D)`` cos/sin → NNTile-graph ``(half,S,B)``
    float32 (C layout)."""
    half = dims.head_size // 2
    cos_half = cos[:, :, :half].to(torch.float32).detach().cpu().numpy()
    sin_half = sin[:, :, :half].to(torch.float32).detach().cpu().numpy()
    # (B, S, half) → (half, S, B)
    cos_np = np.transpose(cos_half, (2, 1, 0))
    sin_np = np.transpose(sin_half, (2, 1, 0))
    return cos_np, sin_np


def _causal_additive_mask_torch(
    batch: int, seq: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """HF additive mask (4D), same construction as ``test_llama_attention``."""
    mask = np.array(np.triu(np.ones((seq, seq))), dtype=bool, order="F")
    mask_torch = torch.tensor(
        np.array(1 - mask, dtype=np.float32),
    ).T * torch.finfo(torch.float32).min
    mask_torch = mask_torch.to(device=device, dtype=torch.float32)
    return mask_torch[None, None, :, :].expand(batch, 1, -1, -1). \
        to(dtype=dtype)


def _sdpa_causal_mask_fortran(seq: int) -> np.ndarray:
    """``(k_seq, q_seq)`` mask for graph ``sdpa_eager`` (1 = keep, 0 = mask).

    Stored as float32 because ``safetensors.numpy.save_file`` upgrades numpy
    bool to F32; the C++ test converts back to BOOL for ``mask_scalar``.
    """
    kk = np.arange(seq, dtype=np.int64)[:, None]
    qq = np.arange(seq, dtype=np.int64)[None, :]
    allowed = (kk <= qq).astype(np.float32)
    return fortran_order(allowed)


# ── Block generators ─────────────────────────────────────────────────----


def generate_attention(
    seed: int,
    dims: TestDims = ATTENTION_MHA_DIMS,
    *,
    use_rope: bool = True,
    use_causal_mask: bool = False,
) -> dict[str, np.ndarray]:
    """HuggingFace path aligned with ``test_llama_attention.generate_inputs``.

    Uses ``ATTENTION_MHA_DIMS`` / ``ATTENTION_GQA_DIMS`` (``single_tile``-style
    shapes). RoPE matches the Python test: first argument to
    ``LlamaRotaryEmbedding`` is ``v_proj.weight``; position ids are drawn with
    ``integers(0, seq, size=(batch, seq))`` (same as ``rng`` there).

    When ``use_rope`` is False, cos/sin are replaced with ones/zeros (identity
    RoPE) in PyTorch, and the same identity tensors are still written as
    ``rope_cos`` / ``rope_sin`` (first half-channels, NNTile layout) so the C++
    graph runs the RoPE op like HuggingFace instead of skipping it with null
    pointers.

    When ``use_causal_mask`` is True, ``attention_mask`` matches
    ``test_llama_attention`` and ``attn_mask`` stores the BOOL causal pattern
    for ``sdpa_eager`` in the graph.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)

    pt = PtAttention(config, layer_idx=0)
    pt.eval()

    data = _attn(pt, "attn", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt

    pos_ids = rng.integers(
        0, dims.seq, size=(dims.batch, dims.seq), dtype=np.int64,
    )
    pos_ids_pt = torch.tensor(pos_ids, device=x_pt.device, dtype=torch.long)
    rotary = LlamaRotaryEmbedding(config, device=x_pt.device)
    cos, sin = rotary(pt.v_proj.weight, pos_ids_pt)
    if not use_rope:
        cos = torch.ones_like(cos)
        sin = torch.zeros_like(sin)
    cos_np, sin_np = _rope_half_from_hf(cos, sin, dims)
    data["rope_cos"] = fortran_order(cos_np)
    data["rope_sin"] = fortran_order(sin_np)
    # NNTile (seq, batch) layout — matches ``rope_sin_cos_from_position_ids``
    pos_nntile = pos_ids.T.astype(np.int64)
    data["position_ids"] = fortran_order_int64(pos_nntile)

    attn_mask_torch: torch.Tensor | None = None
    if use_causal_mask:
        attn_mask_torch = _causal_additive_mask_torch(
            dims.batch, dims.seq, x_pt.device, x_pt.dtype,
        )
        data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)

    out = pt(
        x_pt,
        position_embeddings=(cos, sin),
        attention_mask=attn_mask_torch,
    )[0]
    data["output_ref"] = _out_to_nntile(out)

    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data


def attention_fixture_json_payload(
    stem: str,
    dims: TestDims,
    forward_tol: float,
    backward_tol: float,
) -> dict:
    """Sidecar for C++ graph tests (``version`` must match reader in
    ``llama_attention.cc``)."""
    st_name = f"{stem}.safetensors"
    return {
        "version": 2,
        "stem": stem,
        "safetensors": st_name,
        "sequence_length": dims.seq,
        "batch": dims.batch,
        "llama": {
            "hidden_size": dims.hidden,
            "num_attention_heads": dims.n_heads,
            "num_key_value_heads": dims.kv_heads,
        },
        "tolerances": {
            "forward": forward_tol,
            "backward": backward_tol,
        },
    }


def write_attention_fixture_json(
    out: Path,
    stem: str,
    dims: TestDims,
    forward_tol: float,
    backward_tol: float,
) -> None:
    path = out / f"{stem}.json"
    path.write_text(
        json.dumps(
            attention_fixture_json_payload(
                stem, dims, forward_tol, backward_tol,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {path}")


def mlp_fixture_json_payload(
    stem: str,
    dims: TestDims,
    forward_tol: float,
    backward_tol: float,
) -> dict:
    """Sidecar for C++ graph tests (``version`` must match reader in
    ``llama_mlp.cc``)."""
    st_name = f"{stem}.safetensors"
    return {
        "version": 2,
        "stem": stem,
        "safetensors": st_name,
        "sequence_length": dims.seq,
        "batch": dims.batch,
        "llama": {
            "hidden_size": dims.hidden,
            "intermediate_size": dims.intermediate,
            "num_attention_heads": dims.n_heads,
            "num_key_value_heads": dims.kv_heads,
        },
        "tolerances": {
            "forward": forward_tol,
            "backward": backward_tol,
        },
    }


def write_mlp_fixture_json(
    out: Path,
    stem: str,
    dims: TestDims,
    forward_tol: float,
    backward_tol: float,
) -> None:
    path = out / f"{stem}.json"
    path.write_text(
        json.dumps(
            mlp_fixture_json_payload(stem, dims, forward_tol, backward_tol),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {path}")


def decoder_fixture_json_payload(
    stem: str,
    dims: TestDims,
    forward_tol: float,
    backward_tol: float,
) -> dict:
    """Sidecar for C++ graph ``test_llama_decoder``
    (``version`` must match reader)."""
    st_name = f"{stem}.safetensors"
    return {
        "version": 2,
        "stem": stem,
        "safetensors": st_name,
        "sequence_length": dims.seq,
        "batch": dims.batch,
        "llama": {
            "hidden_size": dims.hidden,
            "intermediate_size": dims.intermediate,
            "num_attention_heads": dims.n_heads,
            "num_key_value_heads": dims.kv_heads,
        },
        "tolerances": {
            "forward": forward_tol,
            "backward": backward_tol,
        },
    }


def write_decoder_fixture_json(
    out: Path,
    stem: str,
    dims: TestDims,
    forward_tol: float,
    backward_tol: float,
) -> None:
    path = out / f"{stem}.json"
    path.write_text(
        json.dumps(
            decoder_fixture_json_payload(
                stem, dims, forward_tol, backward_tol,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {path}")


def model_fixture_json_payload(
    stem: str,
    dims: TestDims,
    forward_tol: float,
    backward_tol: float,
) -> dict:
    """Sidecar for C++ graph ``test_llama_model``
    (``version`` must match reader)."""
    st_name = f"{stem}.safetensors"
    return {
        "version": 2,
        "stem": stem,
        "safetensors": st_name,
        "sequence_length": dims.seq,
        "batch": dims.batch,
        "llama": {
            "vocab_size": dims.vocab,
            "hidden_size": dims.hidden,
            "intermediate_size": dims.intermediate,
            "num_hidden_layers": dims.num_layers,
            "num_attention_heads": dims.n_heads,
            "num_key_value_heads": dims.kv_heads,
        },
        "tolerances": {
            "forward": forward_tol,
            "backward": backward_tol,
        },
    }


def write_model_fixture_json(
    out: Path,
    stem: str,
    dims: TestDims,
    forward_tol: float,
    backward_tol: float,
) -> None:
    path = out / f"{stem}.json"
    path.write_text(
        json.dumps(
            model_fixture_json_payload(stem, dims, forward_tol, backward_tol),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {path}")


def write_causal_fixture_json(
    out: Path,
    stem: str,
    dims: TestDims,
    forward_tol: float,
    backward_tol: float,
) -> None:
    """Same JSON schema as ``write_model_fixture_json``
    (``test_llama_causal``)."""
    path = out / f"{stem}.json"
    path.write_text(
        json.dumps(
            model_fixture_json_payload(stem, dims, forward_tol, backward_tol),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {path}")


def write_attention_rope_mask_variant_files(out: Path, seed: int) -> None:
    """Write extra attn safetensors for RoPE / causal-mask combinations."""
    specs: list[tuple[str, TestDims, bool, bool, float, float]] = [
        ("llama_attention_no_rope", ATTENTION_MHA_DIMS,
         False, False, 1e-6, 1e-6),
        ("llama_attention_causal", ATTENTION_MHA_DIMS,
         True, True, 1e-6, 1e-6),
        ("llama_attention_no_rope_causal", ATTENTION_MHA_DIMS,
         False, True, 1e-6, 1e-6),
        ("llama_attention_gqa_no_rope", ATTENTION_GQA_DIMS,
         False, False, 1e-6, 1e-6),
        ("llama_attention_gqa_causal", ATTENTION_GQA_DIMS,
         True, True, 1e-6, 1e-6),
        ("llama_attention_gqa_no_rope_causal", ATTENTION_GQA_DIMS,
         False, True, 1e-6, 1e-6),
    ]
    for stem, dims, rope, causal, fwd_tol, bwd_tol in specs:
        payload = generate_attention(
            seed, dims, use_rope=rope, use_causal_mask=causal,
        )
        fname = f"{stem}.safetensors"
        path = str(out / fname)
        save_file(payload, path)
        print(f"Saved {path}")
        write_attention_fixture_json(out, stem, dims, fwd_tol, bwd_tol)


def generate_mlp(
    seed: int, dims: TestDims = MHA_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)

    pt = PtMLP(config)
    pt.eval()

    data = _mlp(pt, "mlp")
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt

    out = pt(x_pt)
    data["output_ref"] = _out_to_nntile(out)

    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data


def generate_decoder(
    seed: int, dims: TestDims = DECODER_MHA_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)

    pt = PtDecoderLayer(config, layer_idx=0)
    pt.eval()

    data = _decoder_layer(pt, "decoder", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt

    position_ids = torch.arange(
        dims.seq, device=x_pt.device, dtype=torch.long,
    ).unsqueeze(0).expand(dims.batch, -1)
    rotary = LlamaRotaryEmbedding(config, device=x_pt.device)
    cos, sin = rotary(x_pt, position_ids)
    cos_np, sin_np = _rope_half_from_hf(cos, sin, dims)
    data["rope_cos"] = fortran_order(cos_np)
    data["rope_sin"] = fortran_order(sin_np)
    out = pt(x_pt, position_embeddings=(cos, sin))[0]
    data["output_ref"] = _out_to_nntile(out)

    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data


def generate_model(
    seed: int, dims: TestDims = MODEL_MHA_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)

    pt = PtModel(config)
    pt.eval()

    data = _model_weights(pt, "model", dims)
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt

    inputs_embeds = pt.embed_tokens(ids_pt)
    # Match HF ``LlamaModel`` prefill: ``position_ids`` must include the batch
    # dimension so ``rotary_emb`` returns ``(B,S,...)`` like
    # ``generate_decoder``.
    position_ids = torch.arange(
        0, inputs_embeds.shape[1],
        device=inputs_embeds.device, dtype=torch.long,
    ).unsqueeze(0).expand(inputs_embeds.shape[0], -1)
    cos, sin = pt.rotary_emb(inputs_embeds, position_ids)
    cos_np, sin_np = _rope_half_from_hf(cos, sin, dims)
    data["rope_cos"] = fortran_order(cos_np)
    data["rope_sin"] = fortran_order(sin_np)
    data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)

    out = pt(ids_pt).last_hidden_state
    data["output_ref"] = _out_to_nntile(out)

    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    return data


def generate_causal(
    seed: int, dims: TestDims = CAUSAL_MHA_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)

    pt = PtCausalLM(config)
    pt.eval()

    data = _model_weights(pt.model, "model.model", dims)
    data["model.lm_head.weight"] = _linear(pt.lm_head)
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt

    inputs_embeds = pt.model.embed_tokens(ids_pt)
    position_ids = torch.arange(
        0, inputs_embeds.shape[1],
        device=inputs_embeds.device, dtype=torch.long,
    ).unsqueeze(0).expand(inputs_embeds.shape[0], -1)
    cos, sin = pt.model.rotary_emb(inputs_embeds, position_ids)
    cos_np, sin_np = _rope_half_from_hf(cos, sin, dims)
    data["rope_cos"] = fortran_order(cos_np)
    data["rope_sin"] = fortran_order(sin_np)
    data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)

    out = pt(ids_pt).logits
    data["output_ref"] = _out_to_nntile(out)

    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    return data


GENERATORS = {
    "attention": generate_attention,
    "mlp": generate_mlp,
    "decoder": generate_decoder,
    "model": generate_model,
    "causal": generate_causal,
    "attention_gqa": lambda seed: generate_attention(seed, ATTENTION_GQA_DIMS),
    "decoder_gqa": lambda seed: generate_decoder(seed, DECODER_GQA_DIMS),
    "model_gqa": lambda seed: generate_model(seed, MODEL_GQA_DIMS),
    "causal_gqa": lambda seed: generate_causal(seed, CAUSAL_GQA_DIMS),
}

# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Llama block test data (safetensors)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--block",
        choices=GENERATORS,
        help="Llama block to generate data for",
    )
    mode.add_argument(
        "--write-attention-rope-mask-variants",
        action="store_true",
        help=(
            "Write six extra attention safetensors (MHA/GQA × no-RoPE / "
            "causal / both) for C++ graph tests; does not overwrite default "
            "llama_attention.safetensors or llama_attention_gqa.safetensors."
        ),
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.write_attention_rope_mask_variants:
        write_attention_rope_mask_variant_files(out, args.seed)
        return 0

    data = GENERATORS[args.block](args.seed)

    bundle_path = str(out / f"llama_{args.block}.safetensors")
    save_file(data, bundle_path)
    print(f"Saved {bundle_path}")
    if args.block == "attention":
        write_attention_fixture_json(
            out, "llama_attention", ATTENTION_MHA_DIMS, 1e-6, 1e-6,
        )
    elif args.block == "attention_gqa":
        write_attention_fixture_json(
            out, "llama_attention_gqa", ATTENTION_GQA_DIMS, 1e-6, 1e-6,
        )
    elif args.block == "mlp":
        write_mlp_fixture_json(out, "llama_mlp", MHA_DIMS, 1e-6, 1e-6)
    elif args.block == "decoder":
        write_decoder_fixture_json(
            out, "llama_decoder", DECODER_MHA_DIMS, 1e-6, 1e-6,
        )
    elif args.block == "decoder_gqa":
        write_decoder_fixture_json(
            out, "llama_decoder_gqa", DECODER_GQA_DIMS, 1.3e-6, 1.3e-6,
        )
    elif args.block == "model":
        write_model_fixture_json(
            out, "llama_model", MODEL_MHA_DIMS, 1e-6, 1e-6,
        )
    elif args.block == "model_gqa":
        # GQA full-model float32 vs PyTorch ~1.1e-6 relative Frobenius here.
        write_model_fixture_json(
            out, "llama_model_gqa", MODEL_GQA_DIMS, 1.3e-6, 1.3e-6,
        )
    elif args.block == "causal":
        write_causal_fixture_json(
            out, "llama_causal", CAUSAL_MHA_DIMS, 1e-6, 1e-6,
        )
    elif args.block == "causal_gqa":
        # Same margin as ``model_gqa`` (logits + lm_head).
        write_causal_fixture_json(
            out, "llama_causal_gqa", CAUSAL_GQA_DIMS, 1.3e-6, 1.3e-6,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
