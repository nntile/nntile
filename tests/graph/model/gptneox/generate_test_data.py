#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file tests/graph/model/gptneox/generate_test_data.py
# Generate GPT-NeoX building-block test data in safetensors format.
#
# @version 1.1.0

"""Generate reference test data for NNTile GPT-NeoX graph C++ tests.

For each block the script creates ``gptneox_<block>.safetensors`` plus a paired
``.json`` sidecar (geometry, tolerances) read by the corresponding C++ tests.

Uses **HuggingFace Transformers** (``modeling_gpt_neox``) for forward and
backward references plus NumPy layout wrangling for NNTile safetensors — no
custom attention reimplementation. Weights are split from HF
``query_key_value`` / ``dense`` into graph
``q/k/v/o`` layouts; Q/K get RoPE head-dim interleaving via
``rotate_tensor_in`` (axis 1, ``rotary_pct``), matching
``GPTNeoXAttention.from_torch`` and Llama fixtures. Attention
references call ``GPTNeoXAttention`` with ``_attn_implementation="eager"``,
``GPTNeoXRotaryEmbedding`` cos/sin, and additive ``attention_mask``: causal
upper-triangular when ``use_causal_mask=True``, **zeros** when False (no mask).
``use_rope=False`` uses identity cos/sin in PyTorch and still writes
``rope_cos`` / ``rope_sin`` for the C++ graph (Llama-style). Graph tests load
``attn_mask`` as float32 Fortran ``(seq, seq)`` (1 = keep) for ``sdpa_eager``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.numpy import save_file
from transformers import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention as PtAttention, GPTNeoXForCausalLM as PtCausalLM,
    GPTNeoXLayer as PtLayer, GPTNeoXMLP as PtMLP, GPTNeoXModel as PtModel,
    GPTNeoXRotaryEmbedding)

# ── Test dimension bundles ────────────────────────────────────────────────


@dataclass
class TestDims:
    hidden: int
    intermediate: int
    n_heads: int
    seq: int
    batch: int
    vocab: int
    num_layers: int
    layer_norm_eps: float = 1e-5
    rotary_pct: float = 1.0
    rotary_emb_base: float = 10000.0

    @property
    def head_size(self) -> int:
        return self.hidden // self.n_heads


MLP_DIMS = TestDims(
    hidden=8, intermediate=16, n_heads=4,
    seq=4, batch=2, vocab=100, num_layers=1,
)

ATTENTION_DIMS = TestDims(
    hidden=64, intermediate=256, n_heads=4,
    seq=8, batch=2, vocab=100, num_layers=1,
)

DECODER_DIMS = ATTENTION_DIMS
MODEL_DIMS = ATTENTION_DIMS
CAUSAL_DIMS = ATTENTION_DIMS


def fortran_order(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    return a.ravel("F").reshape(a.shape)


def fortran_order_int64(arr: np.ndarray) -> np.ndarray:
    """int64 on-disk layout: ``ravel('F').reshape`` like ``fortran_order``."""
    a = np.asarray(arr, dtype=np.int64)
    return a.ravel("F").reshape(a.shape)


def _make_config(dims: TestDims) -> GPTNeoXConfig:
    return GPTNeoXConfig(
        vocab_size=dims.vocab,
        hidden_size=dims.hidden,
        num_hidden_layers=dims.num_layers,
        num_attention_heads=dims.n_heads,
        intermediate_size=dims.intermediate,
        max_position_embeddings=max(dims.seq * 2, 128),
        layer_norm_epsilon=dims.layer_norm_eps,
        rotary_pct=dims.rotary_pct,
        rotary_emb_base=dims.rotary_emb_base,
        use_parallel_residual=True,
        attention_bias=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        _attn_implementation="eager",
    )


def _layer_norm(ln, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.gamma": fortran_order(ln.weight.detach().numpy()),
        f"{prefix}.beta": fortran_order(ln.bias.detach().numpy()),
    }


def _rotate_tensor_in_for_rope(
    x: np.ndarray, axis: int, rotary_pct: float,
) -> np.ndarray:
    """Interleave RoPE pairs on ``axis`` (first ``rotary_pct`` of that axis).

    Same layout as ``nntile.layer.GPTNeoXAttention.rotate_tensor_in`` and
    ``kernel::rope`` — HF forward still uses unrotated weights; only the
    safetensors Q/K layouts written for the C++ graph are transformed.
    """
    k_elements = int(x.shape[axis] * rotary_pct)
    if axis == 0:
        new_shape = (1, k_elements, int(np.prod(x.shape[1:])))
    elif axis == x.ndim - 1:
        new_shape = (int(np.prod(x.shape[:-1])), k_elements, 1)
    else:
        new_shape = (
            int(np.prod(x.shape[:axis])),
            k_elements,
            int(np.prod(x.shape[axis + 1:])),
        )
    if axis == 0:
        x_selected = x[:k_elements, ...]
    elif axis == x.ndim - 1:
        x_selected = x[..., :k_elements]
    else:
        slice_obj = [slice(None)] * x.ndim
        slice_obj[axis] = slice(0, k_elements)
        x_selected = x[tuple(slice_obj)]

    x_reshaped = x_selected.reshape(new_shape)
    mid = k_elements // 2
    y_reshaped = np.empty_like(x_reshaped)
    y_reshaped[:, 0::2, :] = x_reshaped[:, :mid, :]
    y_reshaped[:, 1::2, :] = x_reshaped[:, mid:, :]

    result = np.asarray(x, dtype=np.float32).copy()
    if axis == 0:
        result[:k_elements, ...] = y_reshaped.reshape(x_selected.shape)
    elif axis == x.ndim - 1:
        result[..., :k_elements] = y_reshaped.reshape(x_selected.shape)
    else:
        slice_obj = [slice(None)] * x.ndim
        slice_obj[axis] = slice(0, k_elements)
        result[tuple(slice_obj)] = y_reshaped.reshape(x_selected.shape)
    return result


def _gptneox_attn_weights(
    attn: PtAttention, prefix: str, dims: TestDims,
) -> dict[str, np.ndarray]:
    """Map HF ``query_key_value`` + ``dense`` to NNTile graph layouts."""
    H = dims.hidden
    nh = dims.n_heads
    hd = dims.head_size
    qkv_w = attn.query_key_value.weight.detach().numpy()
    qkv = qkv_w.reshape(nh, 3 * hd, H)
    q = _rotate_tensor_in_for_rope(
        np.asarray(qkv[:, :hd, :], dtype=np.float32), 1, dims.rotary_pct,
    )
    k = _rotate_tensor_in_for_rope(
        np.asarray(qkv[:, hd : 2 * hd, :], dtype=np.float32),
        1,
        dims.rotary_pct,
    )
    v = qkv[:, 2 * hd : 3 * hd, :]
    o = attn.dense.weight.detach().numpy().reshape(H, nh, hd)
    return {
        f"{prefix}.q_weight": fortran_order(q),
        f"{prefix}.k_weight": fortran_order(k),
        f"{prefix}.v_weight": fortran_order(v),
        f"{prefix}.o_weight": fortran_order(o),
    }


def _gptneox_mlp_weights(mlp: PtMLP, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.fc1.weight": fortran_order(
            mlp.dense_h_to_4h.weight.detach().numpy().T),
        f"{prefix}.fc1.bias": fortran_order(
            mlp.dense_h_to_4h.bias.detach().numpy()),
        f"{prefix}.fc2.weight": fortran_order(
            mlp.dense_4h_to_h.weight.detach().numpy().T),
        f"{prefix}.fc2.bias": fortran_order(
            mlp.dense_4h_to_h.bias.detach().numpy()),
    }


def _gptneox_decoder_weights(
    layer: PtLayer, prefix: str, dims: TestDims,
) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {}
    d.update(_layer_norm(layer.input_layernorm, f"{prefix}.input_norm"))
    d.update(_gptneox_attn_weights(
        layer.attention, f"{prefix}.attention", dims))
    d.update(_layer_norm(
        layer.post_attention_layernorm, f"{prefix}.post_attn_norm"))
    d.update(_gptneox_mlp_weights(layer.mlp, f"{prefix}.mlp"))
    return d


def _embed(embed, prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}.vocab": fortran_order(embed.weight.detach().numpy().T)}


def _model_weights(
    model: PtModel, prefix: str, dims: TestDims,
) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {}
    d.update(_embed(model.embed_in, f"{prefix}.embed_tokens"))
    d.update(_layer_norm(model.final_layer_norm, f"{prefix}.norm"))
    for i, layer in enumerate(model.layers):
        d.update(_gptneox_decoder_weights(layer, f"{prefix}.layers_{i}", dims))
    return d


def _lm_head_to_linear_weight(lm) -> np.ndarray:
    return fortran_order(lm.weight.detach().numpy().T)


def _hidden_input(rng, dims: TestDims, scale: float = 0.1):
    x = rng.standard_normal(
        (dims.hidden, dims.seq, dims.batch),
    ).astype(np.float32) * scale
    x_nt = fortran_order(x)
    x_pt = torch.tensor(x.transpose(2, 1, 0).copy(), requires_grad=True)
    return x_nt, x_pt


def _grad_output(rng, pt_out: torch.Tensor, scale: float = 0.1):
    g = rng.standard_normal(pt_out.shape).astype(np.float32) * scale
    g_pt = torch.tensor(g)
    g_nt = fortran_order(g.transpose(2, 1, 0))
    return g_nt, g_pt


def _ids_input(rng, dims: TestDims):
    ids = rng.integers(
        0, dims.vocab, size=(dims.seq, dims.batch),
    ).astype(np.int64)
    ids_nt = ids.ravel("F").reshape(ids.shape)
    ids_pt = torch.tensor(ids.T.copy(), dtype=torch.long)
    return ids_nt, ids_pt


def _attention_position_ids(
    dims: TestDims,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    """``0 .. seq-1`` per batch row — matches C++ training/inference defaults.

    Returns ``position_ids`` in NNTile ``(seq, batch)`` Fortran layout and
    PyTorch ``(batch, seq)`` for HF-style helpers.
    """
    pos_pt = torch.arange(
        dims.seq, device=device, dtype=torch.long,
    ).unsqueeze(0).expand(dims.batch, dims.seq)
    pos_nntile = np.asfortranarray(
        pos_pt.detach().cpu().numpy().T.astype(np.int64),
    )
    return pos_nntile, pos_pt


def _out_to_nntile(pt_out: torch.Tensor) -> np.ndarray:
    return fortran_order(pt_out.detach().numpy().transpose(2, 1, 0))


def _sdpa_causal_mask_fortran(seq: int) -> np.ndarray:
    """Causal mask for ``sdpa_eager`` (1 = keep).

    Flat layout ``mask[key + query * seq]`` matches ``test_sdpa_eager`` and
    ``load_attn_mask_bool``. Stored as 1-D ``(seq * seq,)`` so safetensors does
    not permute a 2-D C/F layout.
    """
    flat = np.zeros(seq * seq, dtype=np.float32)
    for qq in range(seq):
        for kk in range(seq):
            flat[kk + qq * seq] = float(kk <= qq)
    return flat


def _causal_additive_mask_torch(
    batch: int, seq: int, device: torch.device,
) -> torch.Tensor:
    mask = np.array(np.triu(np.ones((seq, seq))), dtype=bool, order="F")
    mask_torch = torch.tensor(
        np.array(1 - mask, dtype=np.float32),
    ).T * torch.finfo(torch.float32).min
    mask_torch = mask_torch.to(device=device, dtype=torch.float32)
    return mask_torch[None, None, :, :].expand(batch, 1, -1, -1)


def _gptneox_rope_dim(dims: TestDims) -> int:
    """Match C++ ``gptneox_rope_dim`` (even, from ``rotary_pct``)."""
    pct = dims.rotary_pct
    dim = int(round(dims.head_size * pct))
    if dim < 2:
        dim = 2
    if dim % 2 != 0:
        dim -= 1
    if dim > dims.head_size:
        dim = dims.head_size
        if dim % 2 != 0:
            dim -= 1
    return dim


def _rope_half_from_hf(
    cos: torch.Tensor, sin: torch.Tensor, dims: TestDims,
) -> tuple[np.ndarray, np.ndarray]:
    """HF ``(B,S,D)`` cos/sin → NNTile-graph ``(half,S,B)`` float32."""
    half = _gptneox_rope_dim(dims) // 2
    cos_half = cos[:, :, :half].to(torch.float32).detach().cpu().numpy()
    sin_half = sin[:, :, :half].to(torch.float32).detach().cpu().numpy()
    cos_np = np.transpose(cos_half, (2, 1, 0))
    sin_np = np.transpose(sin_half, (2, 1, 0))
    return cos_np, sin_np


def _zero_additive_attention_mask_torch(
    batch: int, seq: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Additive no-mask for HF eager attention (all zeros, ``[nomask]``)."""
    return torch.zeros(
        batch, 1, seq, seq, device=device, dtype=dtype,
    )


def _gptneox_mlp_forward(mlp: PtMLP, x_pt: torch.Tensor) -> torch.Tensor:
    """HF ``GPTNeoXMLP`` forward (matches graph ``GptneoxMlp`` with biases)."""
    h = mlp.dense_h_to_4h(x_pt)
    h = mlp.act(h)
    return mlp.dense_4h_to_h(h)


def _hf_attention_mask_torch(
    dims: TestDims,
    x_pt: torch.Tensor,
    *,
    use_causal_mask: bool,
) -> torch.Tensor:
    if use_causal_mask:
        return _causal_additive_mask_torch(
            dims.batch, dims.seq, x_pt.device,
        ).to(dtype=x_pt.dtype)
    return _zero_additive_attention_mask_torch(
        dims.batch, dims.seq, x_pt.device, x_pt.dtype,
    )


def _hf_gptneox_attention(
    attn: PtAttention,
    x_pt: torch.Tensor,
    dims: TestDims,
    rotary: GPTNeoXRotaryEmbedding,
    pos_ids_pt: torch.Tensor,
    *,
    use_rope: bool,
    use_causal_mask: bool,
) -> torch.Tensor:
    """HF ``GPTNeoXAttention`` (eager), aligned with Llama fixtures."""
    cos, sin = rotary(x_pt, pos_ids_pt)
    if not use_rope:
        cos = torch.ones_like(cos)
        sin = torch.zeros_like(sin)
    attn_mask = _hf_attention_mask_torch(
        dims, x_pt, use_causal_mask=use_causal_mask,
    )
    return attn(
        x_pt,
        attention_mask=attn_mask,
        position_embeddings=(cos, sin),
    )[0]


def _gptneox_decoder_forward(
    layer: PtLayer,
    x_pt: torch.Tensor,
    rotary: GPTNeoXRotaryEmbedding,
    pos_ids_pt: torch.Tensor,
    dims: TestDims,
    *,
    use_causal_mask: bool = False,
) -> torch.Tensor:
    """HF ``GPTNeoXLayer`` forward (parallel residual when configured)."""
    attn_mask = _hf_attention_mask_torch(
        dims, x_pt, use_causal_mask=use_causal_mask,
    )
    cos, sin = rotary(x_pt, pos_ids_pt)
    return layer(
        x_pt,
        attention_mask=attn_mask,
        position_embeddings=(cos, sin),
    )[0]


def _gptneox_model_forward(
    model: PtModel,
    ids_pt: torch.Tensor,
    dims: TestDims,
    *,
    use_causal_mask: bool = False,
) -> torch.Tensor:
    """HF ``GPTNeoXModel`` forward with additive attention mask."""
    x = model.embed_in(ids_pt)
    pos_ids_pt = torch.arange(
        dims.seq, device=x.device, dtype=torch.long,
    ).unsqueeze(0).expand(dims.batch, dims.seq)
    attn_mask = _hf_attention_mask_torch(
        dims, x, use_causal_mask=use_causal_mask,
    )
    cos, sin = model.rotary_emb(x, pos_ids_pt)
    for layer in model.layers:
        x = layer(
            x,
            attention_mask=attn_mask,
            position_embeddings=(cos, sin),
        )[0]
    return model.final_layer_norm(x)


def _gptneox_fixture_json(
    stem: str,
    dims: TestDims,
    forward_tol: float,
    backward_tol: float,
) -> dict:
    return {
        "version": 2,
        "stem": stem,
        "safetensors": f"{stem}.safetensors",
        "sequence_length": dims.seq,
        "batch": dims.batch,
        "gptneox": {
            "vocab_size": dims.vocab,
            "hidden_size": dims.hidden,
            "intermediate_size": dims.intermediate,
            "num_hidden_layers": dims.num_layers,
            "num_attention_heads": dims.n_heads,
            "head_dim": dims.head_size,
            "max_position_embeddings": max(dims.seq * 2, 128),
            "layer_norm_eps": dims.layer_norm_eps,
            "rotary_pct": dims.rotary_pct,
            "rotary_emb_base": dims.rotary_emb_base,
            "use_parallel_residual": True,
            "attention_bias": False,
        },
        "tolerances": {
            "forward": forward_tol,
            "backward": backward_tol,
        },
    }


def write_fixture_json(
    out: Path,
    stem: str,
    dims: TestDims,
    forward_tol: float,
    backward_tol: float,
) -> None:
    path = out / f"{stem}.json"
    path.write_text(
        json.dumps(
            _gptneox_fixture_json(stem, dims, forward_tol, backward_tol),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {path}")


def write_attention_rope_mask_variant_files(out: Path, seed: int) -> None:
    """Write extra attention safetensors (RoPE / causal-mask variants).

    ``gptneox_attention_causal`` is generated by ``--block attention_causal``
    (CTest ``*_attention_causal_data_setup``), not here, to avoid concurrent
    writes with ``--write-attention-rope-mask-variants``.
    """
    specs: list[tuple[str, bool, bool, float, float]] = [
        ("gptneox_attention_no_rope", False, False, 1e-6, 1e-6),
        ("gptneox_attention_no_rope_causal", False, True, 1e-6, 1e-6),
    ]
    for stem, rope, causal, fwd_tol, bwd_tol in specs:
        payload = generate_attention(
            seed, ATTENTION_DIMS, use_rope=rope, use_causal_mask=causal,
        )
        fname = f"{stem}.safetensors"
        path = str(out / fname)
        save_file(payload, path)
        print(f"Saved {path}")
        write_fixture_json(out, stem, ATTENTION_DIMS, fwd_tol, bwd_tol)


def _write_rope_and_position(
    data: dict[str, np.ndarray],
    pos_nntile: np.ndarray,
    cos_np: np.ndarray,
    sin_np: np.ndarray,
) -> None:
    """Store RoPE inputs in SafeTensors C-order; C++ maps to NNTile Fortran."""
    data["position_ids"] = np.ascontiguousarray(
        pos_nntile, dtype=np.int64,
    )
    data["rope_cos"] = np.ascontiguousarray(cos_np, dtype=np.float32)
    data["rope_sin"] = np.ascontiguousarray(sin_np, dtype=np.float32)


def generate_mlp(
    seed: int, dims: TestDims = MLP_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtMLP(config)
    pt.eval()
    data = _gptneox_mlp_weights(pt, "mlp")
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    out = _gptneox_mlp_forward(pt, x_pt)
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data


def generate_attention(
    seed: int,
    dims: TestDims = ATTENTION_DIMS,
    *,
    use_rope: bool = True,
    use_causal_mask: bool = False,
) -> dict[str, np.ndarray]:
    """HuggingFace ``GPTNeoXAttention`` reference (Llama-style fixtures).

    ``use_rope=False`` replaces cos/sin with ones/zeros in PyTorch but still
    writes ``rope_cos`` / ``rope_sin`` for the C++ graph.

    ``use_causal_mask=False`` uses a zero additive ``attention_mask`` in HF;
    the graph bundle omits ``attn_mask`` (no causal ``sdpa_eager`` mask).

    ``use_causal_mask=True`` uses the causal additive HF mask and stores
    ``attn_mask`` for the graph.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtAttention(config, layer_idx=0)
    pt.eval()
    data = _gptneox_attn_weights(pt, "attn", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt

    pos_nntile, pos_ids_pt = _attention_position_ids(dims, x_pt.device)
    rotary = GPTNeoXRotaryEmbedding(config, device=x_pt.device)
    cos, sin = rotary(x_pt, pos_ids_pt)
    if not use_rope:
        cos = torch.ones_like(cos)
        sin = torch.zeros_like(sin)
    cos_np, sin_np = _rope_half_from_hf(cos, sin, dims)
    _write_rope_and_position(data, pos_nntile, cos_np, sin_np)

    if use_causal_mask:
        data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)

    out = _hf_gptneox_attention(
        pt,
        x_pt,
        dims,
        rotary,
        pos_ids_pt,
        use_rope=use_rope,
        use_causal_mask=use_causal_mask,
    )
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data


def generate_decoder(
    seed: int, dims: TestDims = DECODER_DIMS, *, use_causal_mask: bool = True,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    # Use layer 0 from ``PtModel`` so weights match ``generate_model``
    # (embed init advances the PyTorch RNG before layers are built).
    model = PtModel(config)
    model.eval()
    pt = model.layers[0]
    data = _gptneox_decoder_weights(pt, "decoder", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    pos_nntile, pos_ids_pt = _attention_position_ids(dims, x_pt.device)
    rotary = GPTNeoXRotaryEmbedding(config, device=x_pt.device)
    cos, sin = rotary(x_pt, pos_ids_pt)
    cos_np, sin_np = _rope_half_from_hf(cos, sin, dims)
    _write_rope_and_position(data, pos_nntile, cos_np, sin_np)
    data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
    residual = x_pt
    x_norm = pt.input_layernorm(x_pt)
    data["input_norm_out"] = _out_to_nntile(x_norm)
    attn_out = _hf_gptneox_attention(
        pt.attention,
        x_norm,
        dims,
        rotary,
        pos_ids_pt,
        use_rope=True,
        use_causal_mask=True,
    )
    post_attn = residual + attn_out
    data["post_attn"] = _out_to_nntile(post_attn)
    if pt.use_parallel_residual:
        mlp_in = pt.post_attention_layernorm(residual)
    else:
        mlp_in = pt.post_attention_layernorm(post_attn)
    mlp_out = _gptneox_mlp_forward(pt.mlp, mlp_in)
    data["mlp_out"] = _out_to_nntile(mlp_out)
    out = _gptneox_decoder_forward(
        pt, x_pt, rotary, pos_ids_pt, dims, use_causal_mask=True,
    )
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data


def generate_model(
    seed: int, dims: TestDims = MODEL_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtModel(config)
    pt.eval()
    data = _model_weights(pt, "model", dims)
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt
    data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
    pos_nntile, pos_ids_pt = _attention_position_ids(dims, ids_pt.device)
    hidden = pt.embed_in(ids_pt)
    cos, sin = pt.rotary_emb(hidden, pos_ids_pt)
    cos_np, sin_np = _rope_half_from_hf(cos, sin, dims)
    _write_rope_and_position(data, pos_nntile, cos_np, sin_np)
    out = _gptneox_model_forward(pt, ids_pt, dims, use_causal_mask=True)
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_embed_tokens_vocab"] = fortran_order(
        pt.embed_in.weight.grad.detach().numpy().T)
    return data


def generate_causal(
    seed: int, dims: TestDims = CAUSAL_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtCausalLM(config)
    pt.eval()
    data = _model_weights(pt.gpt_neox, "model.model", dims)
    data["model.lm_head.weight"] = _lm_head_to_linear_weight(pt.embed_out)
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt
    data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
    pos_nntile, pos_ids_pt = _attention_position_ids(dims, ids_pt.device)
    embed = pt.gpt_neox.embed_in(ids_pt)
    cos, sin = pt.gpt_neox.rotary_emb(embed, pos_ids_pt)
    cos_np, sin_np = _rope_half_from_hf(cos, sin, dims)
    _write_rope_and_position(data, pos_nntile, cos_np, sin_np)
    hidden = _gptneox_model_forward(
        pt.gpt_neox, ids_pt, dims, use_causal_mask=True,
    )
    logits = F.linear(hidden, pt.embed_out.weight, None)
    data["output_ref"] = _out_to_nntile(logits)
    g_nt, g_pt = _grad_output(rng, logits)
    logits.backward(g_pt)
    data["grad_output"] = g_nt
    data["grad_embed_tokens_vocab"] = fortran_order(
        pt.gpt_neox.embed_in.weight.grad.detach().numpy().T)
    return data


GENERATORS = {
    "mlp": generate_mlp,
    "attention": lambda seed: generate_attention(seed, use_causal_mask=False),
    "attention_causal": lambda seed: generate_attention(
        seed, use_causal_mask=True,
    ),
    "decoder": generate_decoder,
    "model": generate_model,
    "causal": generate_causal,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate GPT-NeoX block test data (safetensors)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--block",
        choices=GENERATORS,
        help="GPT-NeoX block to generate data for",
    )
    mode.add_argument(
        "--write-attention-rope-mask-variants",
        action="store_true",
        help=(
            "Write two extra attention safetensors (no-RoPE, no-RoPE+causal) "
            "for C++ graph tests; does not overwrite gptneox_attention or "
            "gptneox_attention_causal from --block attention / "
            "attention_causal."
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
    stem = f"gptneox_{args.block}"
    bundle_path = str(out / f"{stem}.safetensors")
    save_file(data, bundle_path)
    print(f"Saved {bundle_path}")

    if args.block == "mlp":
        write_fixture_json(out, stem, MLP_DIMS, 1e-6, 1e-6)
    elif args.block in ("attention", "attention_causal"):
        write_fixture_json(out, stem, ATTENTION_DIMS, 1e-6, 1e-6)
    elif args.block == "decoder":
        write_fixture_json(out, stem, DECODER_DIMS, 1e-6, 1e-6)
    elif args.block in ("model", "causal"):
        write_fixture_json(out, stem, MODEL_DIMS, 1e-6, 1e-6)

    return 0


if __name__ == "__main__":
    sys.exit(main())
