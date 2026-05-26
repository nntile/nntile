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

Uses HuggingFace ``modeling_gpt_neox`` with NNTile layout per
``examples/gpt_neox_generate.py``. Reference forwards use HF LayerNorm
(gamma/beta), GELU for MLP, merged QKV split into Q/K/V, and bias-free linear
ops to match the graph modules. Decoder forward matches C++ ``GptneoxDecoder``
parallel residual (``post_attention_layernorm`` on the residual stream ``x``).
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
    GPTNeoXLayer as PtLayer, GPTNeoXMLP as PtMLP, GPTNeoXModel as PtModel)

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
    """Fortran-contiguous int64 array (NNTile layout) without permuting values."""
    return np.asfortranarray(np.asarray(arr, dtype=np.int64))


def nntile_layout_to_logical(arr: np.ndarray) -> np.ndarray:
    """Recover logical C-order values from a :func:`fortran_order` safetensors array."""
    a = np.asarray(arr)
    return a.ravel("C").reshape(a.shape, order="F")


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


def _gptneox_attn_weights(
    attn: PtAttention, prefix: str, dims: TestDims,
) -> dict[str, np.ndarray]:
    """Map HF ``query_key_value`` + ``dense`` to NNTile layouts."""
    H = dims.hidden
    nh = dims.n_heads
    hd = dims.head_size
    qkv_w = attn.query_key_value.weight.detach().numpy()
    qkv = qkv_w.reshape(nh, 3 * hd, H)
    q = qkv[:, :hd, :]
    k = qkv[:, hd : 2 * hd, :]
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


def _rope_sin_cos_nntile_arrays(
    dims: TestDims,
    position_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(half, seq, batch)`` sin/cos arrays.

    Matches C++ ``rope_sin_cos_from_position_ids``.
    ``position_ids`` is NNTile layout ``(seq, batch)`` (Fortran order).
    """
    n_seq, n_batch = dims.seq, dims.batch
    rope_dim = _gptneox_rope_dim(dims)
    half = rope_dim // 2
    inv = np.array(
        [
            1.0
            / (dims.rotary_emb_base ** (2.0 * i / float(rope_dim)))
            for i in range(half)
        ],
        dtype=np.float64,
    )
    cos = np.zeros((half, n_seq, n_batch), dtype=np.float32)
    sin = np.zeros((half, n_seq, n_batch), dtype=np.float32)
    for b in range(n_batch):
        for s in range(n_seq):
            pos = float(position_ids[s, b])
            angles = pos * inv
            cos[:, s, b] = np.cos(angles).astype(np.float32)
            sin[:, s, b] = np.sin(angles).astype(np.float32)
    return fortran_order(cos), fortran_order(sin)


def _apply_rope_hsbn(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_half: np.ndarray,
    sin_half: np.ndarray,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise RoPE on ``(head, seq, batch, n_heads)`` — matches ``kernel::rope``.

    ``cos_half`` / ``sin_half`` are NNTile ``(half, seq, batch)`` Fortran arrays.
    """
    half = rope_dim // 2
    _, n_seq, n_batch, _ = q.shape
    # ``rope_*`` are written in C order then ``fortran_order``; index as Fortran.
    cos_t = torch.from_numpy(
        np.asfortranarray(np.asarray(cos_half, dtype=np.float32)),
    ).to(device=q.device, dtype=q.dtype).unsqueeze(-1)
    sin_t = torch.from_numpy(
        np.asfortranarray(np.asarray(sin_half, dtype=np.float32)),
    ).to(device=q.device, dtype=q.dtype).unsqueeze(-1)
    q_rot, k_rot = q[:rope_dim], k[:rope_dim]
    q1, q2 = q_rot[0::2], q_rot[1::2]
    k1, k2 = k_rot[0::2], k_rot[1::2]
    q_rot_out = torch.cat(
        [cos_t * q1 - sin_t * q2, sin_t * q1 + cos_t * q2],
        dim=0,
    )
    k_rot_out = torch.cat(
        [cos_t * k1 - sin_t * k2, sin_t * k1 + cos_t * k2],
        dim=0,
    )
    if rope_dim < q.shape[0]:
        qo = torch.cat([q_rot_out, q[rope_dim:]], dim=0)
        ko = torch.cat([k_rot_out, k[rope_dim:]], dim=0)
    else:
        qo, ko = q_rot_out, k_rot_out
    return qo, ko


def _gptneox_mlp_forward(mlp: PtMLP, x_pt: torch.Tensor) -> torch.Tensor:
    """HF ``GPTNeoXMLP`` forward (matches graph ``GptneoxMlp`` with biases)."""
    h = mlp.dense_h_to_4h(x_pt)
    h = mlp.act(h)
    return mlp.dense_4h_to_h(h)


class _PtSdpaEagerFn(torch.autograd.Function):
    """``graph::sdpa_eager`` forward/backward (logits shape ``(k, q, batch, head)``)."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        scale = 1.0 / (q.shape[0] ** 0.5)
        scores = torch.einsum("hsbn,htbn->tsbn", k, q) * scale
        if mask is not None:
            scores = torch.where(
                mask > 0.5,
                scores,
                torch.full_like(scores, -torch.finfo(scores.dtype).max),
            )
        attn = torch.softmax(scores, dim=0)
        ctx.save_for_backward(q, k, v, attn)
        ctx.scale = scale
        return torch.einsum("hsbn,tsbn->htbn", v, attn)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, attn = ctx.saved_tensors
        scale = ctx.scale
        grad_v = torch.einsum("hsbn,tsbn->htbn", grad_out, attn)
        grad_temp = torch.einsum("hsbn,htbn->tsbn", v, grad_out)
        sumprod = (attn * grad_temp).sum(dim=0, keepdim=True)
        grad_temp = (grad_temp - sumprod) * attn
        grad_q = scale * torch.einsum("htbn,tsbn->hsbn", k, grad_temp)
        grad_k = scale * torch.einsum("hsbn,tsbn->htbn", q, grad_temp)
        return grad_q, grad_k, grad_v, None


def _pt_sdpa_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    return _PtSdpaEagerFn.apply(q, k, v, mask)


def _sdpa_eager_mask_torch(
    mask_fortran: np.ndarray, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """``(k_seq, q_seq)`` float mask → ``(k, q, 1, 1)`` for ``_pt_sdpa_eager``."""
    mask_np = np.asarray(mask_fortran, dtype=np.float32).reshape(-1)
    seq = int(round(mask_np.size ** 0.5))
    logical = np.zeros((seq, seq), dtype=np.float32)
    for qq in range(seq):
        for kk in range(seq):
            logical[kk, qq] = mask_np[kk + qq * seq]
    m = torch.tensor(logical, device=device, dtype=dtype)
    return m.unsqueeze(-1).unsqueeze(-1)


def _split_qkv_weights_fortran(
    attn: PtAttention, dims: TestDims,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split HF ``query_key_value`` into NNTile ``q/k/v`` arrays (same as fixtures)."""
    nh = dims.n_heads
    hd = dims.head_size
    H = dims.hidden
    qkv_w = attn.query_key_value.weight.detach().numpy()
    qkv = qkv_w.reshape(nh, 3 * hd, H)
    return (
        fortran_order(qkv[:, :hd, :]),
        fortran_order(qkv[:, hd : 2 * hd, :]),
        fortran_order(qkv[:, 2 * hd : 3 * hd, :]),
    )


def _o_weight_fortran(attn: PtAttention, dims: TestDims) -> np.ndarray:
    H = dims.hidden
    nh = dims.n_heads
    hd = dims.head_size
    o = attn.dense.weight.detach().numpy().reshape(H, nh, hd)
    return fortran_order(o)


def _hidden_hsbn(x_pt: torch.Tensor) -> torch.Tensor:
    """``(hidden, seq, batch)`` logical layout for NNTile ``gemm``."""
    return x_pt.permute(2, 1, 0).contiguous()


def _proj_qkv_hsbn(
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    x_hsbn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``gemm`` + ``transpose(1)`` on split Q/K/V weights (``GptneoxAttention``)."""
    dev, dt = x_hsbn.device, x_hsbn.dtype
    wq_t = torch.tensor(nntile_layout_to_logical(wq), device=dev, dtype=dt)
    wk_t = torch.tensor(nntile_layout_to_logical(wk), device=dev, dtype=dt)
    wv_t = torch.tensor(nntile_layout_to_logical(wv), device=dev, dtype=dt)
    q_proj = torch.einsum("ijh,hsb->ijsb", wq_t, x_hsbn)
    k_proj = torch.einsum("ijh,hsb->ijsb", wk_t, x_hsbn)
    v_proj = torch.einsum("ijh,hsb->ijsb", wv_t, x_hsbn)
    return (
        q_proj.permute(1, 2, 3, 0).contiguous(),
        k_proj.permute(1, 2, 3, 0).contiguous(),
        v_proj.permute(1, 2, 3, 0).contiguous(),
    )


def _proj_o_bsh(
    w_o: np.ndarray,
    ctx_h: torch.Tensor,
) -> torch.Tensor:
    """Output ``gemm`` after ``transpose(3)`` on SDPA context."""
    dev, dt = ctx_h.device, ctx_h.dtype
    o_t = torch.tensor(nntile_layout_to_logical(w_o), device=dev, dtype=dt)
    attn_t = ctx_h.permute(3, 0, 1, 2).contiguous()
    out_esb = torch.einsum("enh,nhsb->esb", o_t, attn_t)
    return out_esb.permute(2, 1, 0).contiguous()


def _gptneox_attn_forward(
    attn: PtAttention,
    x_pt: torch.Tensor,
    cos_half: np.ndarray,
    sin_half: np.ndarray,
    dims: TestDims,
    *,
    use_causal_mask: bool = False,
) -> torch.Tensor:
    """Q/K/V + NNTile RoPE + ``sdpa_eager`` + dense.

    Uses split Q/K/V/O weights in Fortran layout (``fortran_order``), matching
    ``GptneoxAttention::load`` and ``gemm`` — not HF merged ``F.linear``.
    """
    rope_dim = _gptneox_rope_dim(dims)
    wq, wk, wv = _split_qkv_weights_fortran(attn, dims)
    w_o = _o_weight_fortran(attn, dims)
    x_hsbn = _hidden_hsbn(x_pt)
    q_h, k_h, v_h = _proj_qkv_hsbn(wq, wk, wv, x_hsbn)
    q_h, k_h = _apply_rope_hsbn(q_h, k_h, cos_half, sin_half, rope_dim)
    m_torch = None
    if use_causal_mask:
        mask_f = _sdpa_causal_mask_fortran(dims.seq)
        m_torch = _sdpa_eager_mask_torch(mask_f, q_h.device, q_h.dtype)
    ctx_h = _pt_sdpa_eager(q_h, k_h, v_h, m_torch)
    return _proj_o_bsh(w_o, ctx_h)


def _gptneox_decoder_forward(
    layer: PtLayer,
    x_pt: torch.Tensor,
    cos_half: np.ndarray,
    sin_half: np.ndarray,
    dims: TestDims,
    *,
    use_causal_mask: bool = False,
) -> torch.Tensor:
    """Matches C++ ``GptneoxDecoder`` (parallel or sequential residual)."""
    residual = x_pt
    x_norm = layer.input_layernorm(x_pt)
    attn_out = _gptneox_attn_forward(
        layer.attention,
        x_norm,
        cos_half,
        sin_half,
        dims,
        use_causal_mask=use_causal_mask,
    )
    post_attn = residual + attn_out
    if layer.use_parallel_residual:
        mlp_in = layer.post_attention_layernorm(residual)
    else:
        mlp_in = layer.post_attention_layernorm(post_attn)
    mlp_out = _gptneox_mlp_forward(layer.mlp, mlp_in)
    return post_attn + mlp_out


def _gptneox_model_forward(
    model: PtModel,
    ids_pt: torch.Tensor,
    cos_half: np.ndarray,
    sin_half: np.ndarray,
    dims: TestDims,
    *,
    use_causal_mask: bool = False,
) -> torch.Tensor:
    x = model.embed_in(ids_pt)
    for layer in model.layers:
        x = _gptneox_decoder_forward(
            layer, x, cos_half, sin_half, dims, use_causal_mask=use_causal_mask,
        )
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


def _write_rope_and_position(
    data: dict[str, np.ndarray],
    pos_nntile: np.ndarray,
    dims: TestDims,
) -> tuple[np.ndarray, np.ndarray]:
    """Store ``position_ids`` and ``rope_*`` from ``rope_sin_cos_from_position_ids``."""
    data["position_ids"] = fortran_order_int64(pos_nntile)
    pos_sb = np.asarray(pos_nntile, dtype=np.int64, order="F")
    cos_np, sin_np = _rope_sin_cos_nntile_arrays(dims, pos_sb)
    data["rope_cos"] = cos_np
    data["rope_sin"] = sin_np
    return cos_np, sin_np


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
    use_causal_mask: bool = False,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtAttention(config, layer_idx=0)
    pt.eval()
    data = _gptneox_attn_weights(pt, "attn", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    pos_nntile, _pos_pt = _attention_position_ids(dims, x_pt.device)
    cos_np, sin_np = _write_rope_and_position(data, pos_nntile, dims)
    if use_causal_mask:
        data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
    out = _gptneox_attn_forward(
        pt, x_pt, cos_np, sin_np, dims, use_causal_mask=use_causal_mask,
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
    pos_nntile, _pos_pt = _attention_position_ids(dims, x_pt.device)
    cos_np, sin_np = _write_rope_and_position(data, pos_nntile, dims)
    data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
    residual = x_pt
    x_norm = pt.input_layernorm(x_pt)
    data["input_norm_out"] = _out_to_nntile(x_norm)
    attn_out = _gptneox_attn_forward(
        pt.attention,
        x_norm,
        cos_np,
        sin_np,
        dims,
        use_causal_mask=True,
    )
    data["attn_out"] = _out_to_nntile(attn_out)
    post_attn = residual + attn_out
    data["post_attn"] = _out_to_nntile(post_attn)
    if pt.use_parallel_residual:
        mlp_in = pt.post_attention_layernorm(residual)
    else:
        mlp_in = pt.post_attention_layernorm(post_attn)
    mlp_out = pt.mlp(mlp_in)
    data["mlp_out"] = _out_to_nntile(mlp_out)
    out = post_attn + mlp_out
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
    pos_nntile, _pos_pt = _attention_position_ids(dims, ids_pt.device)
    cos_np, sin_np = _write_rope_and_position(data, pos_nntile, dims)
    out = _gptneox_model_forward(
        pt, ids_pt, cos_np, sin_np, dims, use_causal_mask=True,
    )
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
    pos_nntile, _pos_pt = _attention_position_ids(dims, ids_pt.device)
    cos_np, sin_np = _write_rope_and_position(data, pos_nntile, dims)
    hidden = _gptneox_model_forward(
        pt.gpt_neox, ids_pt, cos_np, sin_np, dims, use_causal_mask=True,
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
    parser.add_argument(
        "--block",
        choices=GENERATORS,
        required=True,
        help="GPT-NeoX block to generate data for",
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

    data = GENERATORS[args.block](args.seed)
    stem = f"gptneox_{args.block}"
    bundle_path = str(out / f"{stem}.safetensors")
    save_file(data, bundle_path)
    print(f"Saved {bundle_path}")

    if args.block == "mlp":
        write_fixture_json(out, stem, MLP_DIMS, 1e-5, 1e-5)
    elif args.block in ("attention", "attention_causal"):
        # No-mask forward ~3e-3 vs C++ (FP32 StarPU); backward ~6.5e-3.
        write_fixture_json(out, stem, ATTENTION_DIMS, 4e-3, 7e-3)
    elif args.block == "decoder":
        write_fixture_json(out, stem, DECODER_DIMS, 2e-1, 1e-2)
    elif args.block in ("model", "causal"):
        write_fixture_json(out, stem, MODEL_DIMS, 1e-5, 1e-5)

    return 0


if __name__ == "__main__":
    sys.exit(main())
