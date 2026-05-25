#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file tests/graph/model/t5/generate_test_data.py
# Generate T5 building-block test data in safetensors format.
#
# @version 1.1.0

"""Generate reference test data for NNTile T5 graph C++ tests.

Reference forwards/backwards mirror the C++ graph API (RMSNorm, gated GELU,
``sdpa_eager`` layout, HF→NNTile weight layouts from ``examples/t5_generate.py``).
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
from transformers import T5Config
from transformers.models.t5.modeling_t5 import (
    T5Attention as PtAttention,
    T5Block as PtBlock,
    T5ForConditionalGeneration as PtConditional,
    T5LayerFF as PtLayerFF,
    T5Model as PtModel,
    T5Stack as PtStack,
)

# ── Test dimension bundles ────────────────────────────────────────────────


@dataclass
class TestDims:
    d_model: int
    d_ff: int
    n_heads: int
    d_kv: int
    seq: int
    batch: int
    vocab: int
    num_layers: int
    num_decoder_layers: int
    enc_seq: int | None = None
    dec_seq: int | None = None
    layer_norm_eps: float = 1e-5

    @property
    def head_size(self) -> int:
        return self.d_kv

    @property
    def encoder_seq(self) -> int:
        return self.enc_seq if self.enc_seq is not None else self.seq

    @property
    def decoder_seq(self) -> int:
        return self.dec_seq if self.dec_seq is not None else self.seq


FF_DIMS = TestDims(
    d_model=8, d_ff=16, n_heads=4, d_kv=2,
    seq=4, batch=2, vocab=100, num_layers=1, num_decoder_layers=1,
)

ATTN_DIMS = TestDims(
    d_model=64, d_ff=256, n_heads=4, d_kv=16,
    seq=8, batch=2, vocab=100, num_layers=1, num_decoder_layers=1,
)

CROSS_DIMS = TestDims(
    d_model=64, d_ff=256, n_heads=4, d_kv=16,
    seq=8, batch=2, vocab=100, num_layers=1, num_decoder_layers=1,
    enc_seq=8, dec_seq=6,
)

ENCODER_BLOCK_DIMS = ATTN_DIMS
DECODER_BLOCK_DIMS = CROSS_DIMS
MODEL_DIMS = CROSS_DIMS
CONDITIONAL_DIMS = CROSS_DIMS


def fortran_order(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    return a.ravel("F").reshape(a.shape)


def fortran_order_int64(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.int64)
    return a.ravel("F").reshape(a.shape)


def _make_config(dims: TestDims) -> T5Config:
    return T5Config(
        vocab_size=dims.vocab,
        d_model=dims.d_model,
        d_kv=dims.d_kv,
        d_ff=dims.d_ff,
        num_heads=dims.n_heads,
        num_layers=dims.num_layers,
        num_decoder_layers=dims.num_decoder_layers,
        layer_norm_epsilon=dims.layer_norm_eps,
        dropout_rate=0.0,
        is_gated_act=True,
        feed_forward_proj="gated-gelu",
        _attn_implementation="eager",
    )


def _linear_weight(linear: torch.nn.Linear) -> np.ndarray:
    return fortran_order(linear.weight.detach().numpy().T)


def _rms_gamma(layer_norm: torch.nn.Module, prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}.gamma": fortran_order(layer_norm.weight.detach().numpy())}


def _t5_attn_weights(
    attn: PtAttention, prefix: str, dims: TestDims,
) -> dict[str, np.ndarray]:
    nh, hs, dm = dims.n_heads, dims.head_size, dims.d_model
    q = attn.q.weight.detach().numpy().reshape(nh, hs, dm)
    k = attn.k.weight.detach().numpy().reshape(nh, hs, dm)
    v = attn.v.weight.detach().numpy().reshape(nh, hs, dm)
    o = attn.o.weight.detach().numpy().reshape(dm, nh, hs)
    return {
        f"{prefix}.q_weight": fortran_order(q),
        f"{prefix}.k_weight": fortran_order(k),
        f"{prefix}.v_weight": fortran_order(v),
        f"{prefix}.o_weight": fortran_order(o),
    }


def _t5_ff_weights(ff: PtLayerFF, prefix: str) -> dict[str, np.ndarray]:
    dense = ff.DenseReluDense
    d: dict[str, np.ndarray] = {}
    d.update(_rms_gamma(ff.layer_norm, f"{prefix}.layer_norm"))
    d[f"{prefix}.dense.gate_proj.weight"] = _linear_weight(dense.wi_0)
    d[f"{prefix}.dense.up_proj.weight"] = _linear_weight(dense.wi_1)
    d[f"{prefix}.dense.down_proj.weight"] = _linear_weight(dense.wo)
    return d


def _encoder_block_weights(block: PtBlock, prefix: str, dims: TestDims) -> dict:
    d: dict[str, np.ndarray] = {}
    d.update(_rms_gamma(block.layer[0].layer_norm, f"{prefix}.layer_norm_0"))
    d.update(_t5_attn_weights(block.layer[0].SelfAttention, f"{prefix}.self_attn", dims))
    d.update(_t5_ff_weights(block.layer[1], f"{prefix}.ff"))
    return d


def _decoder_block_weights(block: PtBlock, prefix: str, dims: TestDims) -> dict:
    d: dict[str, np.ndarray] = {}
    d.update(_rms_gamma(block.layer[0].layer_norm, f"{prefix}.layer_norm_0"))
    d.update(_t5_attn_weights(block.layer[0].SelfAttention, f"{prefix}.self_attn", dims))
    d.update(_rms_gamma(block.layer[1].layer_norm, f"{prefix}.layer_norm_1"))
    d.update(
        _t5_attn_weights(
            block.layer[1].EncDecAttention, f"{prefix}.cross_attn", dims,
        ),
    )
    d.update(_t5_ff_weights(block.layer[2], f"{prefix}.ff"))
    return d


def _embed_weights(embed: torch.nn.Embedding, prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}.vocab": fortran_order(embed.weight.detach().numpy().T)}


def _model_weights(model: PtModel, prefix: str, dims: TestDims) -> dict:
    d: dict[str, np.ndarray] = {}
    d.update(_embed_weights(model.shared, f"{prefix}.embed_tokens"))
    d.update(_rms_gamma(model.encoder.final_layer_norm, f"{prefix}.encoder_final_norm"))
    d.update(_rms_gamma(model.decoder.final_layer_norm, f"{prefix}.decoder_final_norm"))
    for i, layer in enumerate(model.encoder.block):
        d.update(_encoder_block_weights(layer, f"{prefix}.encoder_layers_{i}", dims))
    for i, layer in enumerate(model.decoder.block):
        d.update(_decoder_block_weights(layer, f"{prefix}.decoder_layers_{i}", dims))
    return d


def _conditional_weights(model: PtConditional, prefix: str, dims: TestDims) -> dict:
    d = _model_weights(model, f"{prefix}.model", dims)
    d[f"{prefix}.lm_head.weight"] = fortran_order(
        model.lm_head.weight.detach().numpy().T,
    )
    return d


# ── NNTile-graph reference ops (match C++) ───────────────────────────────


def _torch_from_fortran(x_nt: np.ndarray) -> torch.Tensor:
    """NNTile ``(D,S,B)`` Fortran → PyTorch ``(B,S,D)``."""
    x = np.array(x_nt, dtype=np.float32, order="F")
    return torch.tensor(x.transpose(2, 1, 0).copy())


def _weight_from_fortran(w_nt: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.array(w_nt, dtype=np.float32, order="F"))


def _cyclic_transpose(x: torch.Tensor, ndim: int) -> torch.Tensor:
    n = x.dim()
    perm = [(i + ndim) % n for i in range(n)]
    return x.permute(perm).contiguous()


def _gemm_ndim1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``gemm(a,b, ndim=1, batch_ndim=0)`` for 3D tensors."""
    return torch.einsum("...ka,...db->...kb", a, b)


def _gemm_ndim2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``gemm(a,b, ndim=2, batch_ndim=0)`` for 4D ``b``."""
    return torch.einsum("...xy,...xydb->...db", a, b)


def _pt_rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * rms * gamma


def _pt_gated_mlp(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    gate = x @ gate_w
    up = x @ up_w
    hidden = F.gelu(gate) * up
    return hidden @ down_w


def _pt_sdpa_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Match ``graph::sdpa_eager`` (scale = 1/sqrt(q.shape[0]))."""
    scale = 1.0 / (q.shape[0] ** 0.5)
    scores = torch.einsum("hsbn,htbn->stbn", k, q) * scale
    if mask is not None:
        scores = torch.where(
            mask > 0.5,
            scores,
            torch.full_like(scores, -torch.finfo(scores.dtype).max),
        )
    attn = torch.softmax(scores, dim=0)
    return torch.einsum("hsbn,stbn->htbn", v, attn)


def _pt_t5_attention(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    encoder: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    k_src = encoder if encoder is not None else x
    x_dsb = x.permute(2, 1, 0).contiguous()
    k_dsb = k_src.permute(2, 1, 0).contiguous()
    q_proj = torch.einsum("hkd,dsb->hksb", w_q, x_dsb)
    k_proj = torch.einsum("hkd,dsb->hksb", w_k, k_dsb)
    v_proj = torch.einsum("hkd,dsb->hksb", w_v, k_dsb)
    q = _cyclic_transpose(q_proj, 1)
    k = _cyclic_transpose(k_proj, 1)
    v = _cyclic_transpose(v_proj, 1)
    if mask is not None:
        # mask (k_seq, q_seq) float → (k_seq, q_seq, 1, 1)
        m = mask.unsqueeze(-1).unsqueeze(-1)
    else:
        m = None
    attn_out = _pt_sdpa_eager(q, k, v, m)
    attn_t = _cyclic_transpose(attn_out, 3)
    out = torch.einsum("dnh,nhsb->dsb", w_o, attn_t)
    return out.permute(2, 1, 0).contiguous()


def _pt_t5_ff(
    x: torch.Tensor,
    gamma: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_norm = _pt_rms_norm(x, gamma, eps)
    ff = _pt_gated_mlp(x_norm, gate_w, up_w, down_w)
    return x + ff


def _hidden_input(rng, dims: TestDims, *, seq: int | None = None, scale: float = 0.1):
    s = dims.seq if seq is None else seq
    x = rng.standard_normal((dims.d_model, s, dims.batch)).astype(np.float32) * scale
    x_nt = fortran_order(x)
    x_pt = _torch_from_fortran(x_nt).requires_grad_(True)
    return x_nt, x_pt


def _grad_output(rng, pt_out: torch.Tensor, scale: float = 0.1):
    g = rng.standard_normal(pt_out.shape).astype(np.float32) * scale
    g_pt = torch.tensor(g)
    g_nt = fortran_order(g.transpose(2, 1, 0))
    return g_nt, g_pt


def _ids_input(rng, dims: TestDims, *, seq: int | None = None):
    s = dims.encoder_seq if seq == "enc" else (
        dims.decoder_seq if seq == "dec" else dims.seq
    )
    ids = rng.integers(0, dims.vocab, size=(s, dims.batch)).astype(np.int64)
    ids_nt = fortran_order_int64(ids)
    ids_pt = torch.tensor(ids.T.copy(), dtype=torch.long)
    return ids_nt, ids_pt


def _out_to_nntile(pt_out: torch.Tensor) -> np.ndarray:
    return fortran_order(pt_out.detach().numpy().transpose(2, 1, 0))


def _sdpa_causal_mask_fortran(seq: int) -> np.ndarray:
    kk = np.arange(seq, dtype=np.int64)[:, None]
    qq = np.arange(seq, dtype=np.int64)[None, :]
    return fortran_order((kk <= qq).astype(np.float32))


def _cross_attn_mask_fortran(enc_seq: int, dec_seq: int) -> np.ndarray:
    """``(k_seq, q_seq)`` = ``(enc_seq, dec_seq)`` for ``sdpa_eager``."""
    return fortran_order(np.ones((enc_seq, dec_seq), dtype=np.float32))


def _load_block_weights(data: dict, prefix: str) -> dict[str, torch.Tensor]:
    return {k: _weight_from_fortran(v) for k, v in data.items() if k.startswith(prefix)}


def _pt_t5_ff_from_data(
    x_pt: torch.Tensor,
    data: dict[str, np.ndarray],
    prefix: str,
    eps: float,
) -> torch.Tensor:
    gamma = _weight_from_fortran(data[f"{prefix}.layer_norm.gamma"])
    gate_w = _weight_from_fortran(data[f"{prefix}.dense.gate_proj.weight"])
    up_w = _weight_from_fortran(data[f"{prefix}.dense.up_proj.weight"])
    down_w = _weight_from_fortran(data[f"{prefix}.dense.down_proj.weight"])
    return _pt_t5_ff(x_pt, gamma, gate_w, up_w, down_w, eps)


def _pt_t5_attn_from_data(
    x_pt: torch.Tensor,
    data: dict[str, np.ndarray],
    prefix: str,
    encoder_pt: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    w_q = _weight_from_fortran(data[f"{prefix}.q_weight"])
    w_k = _weight_from_fortran(data[f"{prefix}.k_weight"])
    w_v = _weight_from_fortran(data[f"{prefix}.v_weight"])
    w_o = _weight_from_fortran(data[f"{prefix}.o_weight"])
    return _pt_t5_attention(x_pt, w_q, w_k, w_v, w_o, encoder_pt, mask)


def _pt_encoder_block(
    x_pt: torch.Tensor,
    data: dict[str, np.ndarray],
    prefix: str,
    dims: TestDims,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    eps = dims.layer_norm_eps
    gamma0 = _weight_from_fortran(data[f"{prefix}.layer_norm_0.gamma"])
    x_norm = _pt_rms_norm(x_pt, gamma0, eps)
    attn = _pt_t5_attn_from_data(x_norm, data, f"{prefix}.self_attn", mask=mask)
    post = x_pt + attn
    return _pt_t5_ff_from_data(post, data, f"{prefix}.ff", eps)


def _pt_decoder_block(
    x_pt: torch.Tensor,
    enc_pt: torch.Tensor,
    data: dict[str, np.ndarray],
    prefix: str,
    dims: TestDims,
    self_mask: torch.Tensor | None,
    cross_mask: torch.Tensor | None,
) -> torch.Tensor:
    eps = dims.layer_norm_eps
    g0 = _weight_from_fortran(data[f"{prefix}.layer_norm_0.gamma"])
    x_norm = _pt_rms_norm(x_pt, g0, eps)
    self_out = _pt_t5_attn_from_data(
        x_norm, data, f"{prefix}.self_attn", mask=self_mask,
    )
    post_self = x_pt + self_out
    g1 = _weight_from_fortran(data[f"{prefix}.layer_norm_1.gamma"])
    x_norm1 = _pt_rms_norm(post_self, g1, eps)
    cross_out = _pt_t5_attn_from_data(
        x_norm1, data, f"{prefix}.cross_attn", enc_pt, cross_mask,
    )
    post_cross = post_self + cross_out
    return _pt_t5_ff_from_data(post_cross, data, f"{prefix}.ff", eps)


def _pt_t5_model(
    enc_ids: torch.Tensor,
    dec_ids: torch.Tensor,
    data: dict[str, np.ndarray],
    prefix: str,
    dims: TestDims,
    dec_mask: torch.Tensor | None,
    cross_mask: torch.Tensor | None,
    vocab: torch.Tensor | None = None,
) -> torch.Tensor:
    if vocab is None:
        vocab = _weight_from_fortran(
            data[f"{prefix}.embed_tokens.vocab"],
        ).T.requires_grad_(True)
    enc_x = F.embedding(enc_ids, vocab)
    dec_x = F.embedding(dec_ids, vocab)
    hidden = enc_x
    for i in range(dims.num_layers):
        hidden = _pt_encoder_block(
            hidden, data, f"{prefix}.encoder_layers_{i}", dims, None,
        )
    g_enc = _weight_from_fortran(data[f"{prefix}.encoder_final_norm.gamma"])
    enc_states = _pt_rms_norm(hidden, g_enc, dims.layer_norm_eps)
    dec_hidden = dec_x
    for i in range(dims.num_decoder_layers):
        dec_hidden = _pt_decoder_block(
            dec_hidden, enc_states, data,
            f"{prefix}.decoder_layers_{i}", dims, dec_mask, cross_mask,
        )
    g_dec = _weight_from_fortran(data[f"{prefix}.decoder_final_norm.gamma"])
    out = _pt_rms_norm(dec_hidden, g_dec, dims.layer_norm_eps)
    return out.permute(2, 1, 0).contiguous()


def _t5_fixture_json(
    stem: str,
    dims: TestDims,
    forward_tol: float,
    backward_tol: float,
    *,
    enc_seq: int | None = None,
    dec_seq: int | None = None,
) -> dict:
    j: dict = {
        "version": 2,
        "stem": stem,
        "safetensors": f"{stem}.safetensors",
        "sequence_length": dims.seq,
        "batch": dims.batch,
        "t5": {
            "vocab_size": dims.vocab,
            "d_model": dims.d_model,
            "d_kv": dims.d_kv,
            "d_ff": dims.d_ff,
            "num_heads": dims.n_heads,
            "num_layers": dims.num_layers,
            "num_decoder_layers": dims.num_decoder_layers,
            "layer_norm_epsilon": dims.layer_norm_eps,
        },
        "tolerances": {"forward": forward_tol, "backward": backward_tol},
    }
    if enc_seq is not None:
        j["encoder_sequence_length"] = enc_seq
    if dec_seq is not None:
        j["decoder_sequence_length"] = dec_seq
    return j


def write_fixture_json(
    out: Path, stem: str, dims: TestDims, forward_tol: float, backward_tol: float,
    *,
    enc_seq: int | None = None,
    dec_seq: int | None = None,
) -> None:
    path = out / f"{stem}.json"
    path.write_text(
        json.dumps(
            _t5_fixture_json(
                stem, dims, forward_tol, backward_tol,
                enc_seq=enc_seq, dec_seq=dec_seq,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {path}")


def _run_fwd_bwd(
    fwd_fn, x_pt: torch.Tensor, rng, data: dict,
) -> tuple[dict[str, np.ndarray], torch.Tensor]:
    out = fwd_fn(x_pt)
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data, out


def generate_ff(seed: int, dims: TestDims = FF_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtLayerFF(config)
    pt.eval()
    data = _t5_ff_weights(pt, "ff")
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    _run_fwd_bwd(
        lambda x: _pt_t5_ff_from_data(x, data, "ff", dims.layer_norm_eps),
        x_pt, rng, data,
    )
    return data


def generate_attention(
    seed: int,
    dims: TestDims = ATTN_DIMS,
    *,
    causal: bool = False,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtAttention(config, has_relative_attention_bias=False)
    pt.eval()
    data = _t5_attn_weights(pt, "attn", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    mask_pt = None
    if causal:
        data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
        m = torch.tensor(data["attn_mask"].reshape(dims.seq, dims.seq))
        mask_pt = m
    _run_fwd_bwd(
        lambda x: _pt_t5_attn_from_data(x, data, "attn", mask=mask_pt),
        x_pt, rng, data,
    )
    return data


def generate_cross_attention(
    seed: int, dims: TestDims = CROSS_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt_model = PtModel(config)
    pt = pt_model.decoder.block[0].layer[1].EncDecAttention
    pt.eval()
    data = _t5_attn_weights(pt, "cross_attn", dims)
    x_nt, x_pt = _hidden_input(rng, dims, seq=dims.decoder_seq)
    enc_nt, enc_pt = _hidden_input(rng, dims, seq=dims.encoder_seq)
    data["input"] = x_nt
    data["encoder_input"] = enc_nt
    data["cross_attn_mask"] = _cross_attn_mask_fortran(
        dims.encoder_seq, dims.decoder_seq,
    )
    m = torch.tensor(
        data["cross_attn_mask"].reshape(dims.encoder_seq, dims.decoder_seq),
    )

    def fwd(x):
        return _pt_t5_attn_from_data(x, data, "cross_attn", enc_pt, m)

    _run_fwd_bwd(fwd, x_pt, rng, data)
    return data


def generate_encoder_block(
    seed: int, dims: TestDims = ENCODER_BLOCK_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt_model = PtModel(config)
    block = pt_model.encoder.block[0]
    block.eval()
    data = _encoder_block_weights(block, "encoder", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    _run_fwd_bwd(
        lambda x: _pt_encoder_block(x, data, "encoder", dims, None),
        x_pt, rng, data,
    )
    return data


def generate_decoder_block(
    seed: int, dims: TestDims = DECODER_BLOCK_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt_model = PtModel(config)
    block = pt_model.decoder.block[0]
    block.eval()
    data = _decoder_block_weights(block, "decoder", dims)
    x_nt, x_pt = _hidden_input(rng, dims, seq=dims.decoder_seq)
    enc_nt, enc_pt = _hidden_input(rng, dims, seq=dims.encoder_seq)
    data["input"] = x_nt
    data["encoder_hidden_states"] = enc_nt
    data["decoder_attn_mask"] = _sdpa_causal_mask_fortran(dims.decoder_seq)
    data["cross_attn_mask"] = _cross_attn_mask_fortran(
        dims.encoder_seq, dims.decoder_seq,
    )
    dec_m = torch.tensor(
        data["decoder_attn_mask"].reshape(dims.decoder_seq, dims.decoder_seq),
    )
    cross_m = torch.tensor(
        data["cross_attn_mask"].reshape(dims.encoder_seq, dims.decoder_seq),
    )

    def fwd(x):
        return _pt_decoder_block(x, enc_pt, data, "decoder", dims, dec_m, cross_m)

    _run_fwd_bwd(fwd, x_pt, rng, data)
    return data


def generate_model(seed: int, dims: TestDims = MODEL_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtModel(config)
    pt.eval()
    data = _model_weights(pt, "model", dims)
    enc_nt, enc_ids = _ids_input(rng, dims, seq="enc")
    dec_nt, dec_ids = _ids_input(rng, dims, seq="dec")
    data["encoder_input_ids"] = enc_nt
    data["decoder_input_ids"] = dec_nt
    data["decoder_attention_mask"] = _sdpa_causal_mask_fortran(dims.decoder_seq)
    data["cross_attention_mask"] = _cross_attn_mask_fortran(
        dims.encoder_seq, dims.decoder_seq,
    )
    dec_m = torch.tensor(
        data["decoder_attention_mask"].reshape(
            dims.decoder_seq, dims.decoder_seq,
        ),
    )
    cross_m = torch.tensor(
        data["cross_attention_mask"].reshape(
            dims.encoder_seq, dims.decoder_seq,
        ),
    )
    vocab = _weight_from_fortran(
        data["model.embed_tokens.vocab"],
    ).T.requires_grad_(True)
    out = _pt_t5_model(
        enc_ids, dec_ids, data, "model", dims, dec_m, cross_m, vocab=vocab,
    )
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_embed_tokens_vocab"] = fortran_order(
        vocab.grad.detach().numpy().T,
    )
    return data


def generate_conditional(
    seed: int, dims: TestDims = CONDITIONAL_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtConditional(config)
    pt.eval()
    data = _conditional_weights(pt, "conditional", dims)
    enc_nt, enc_ids = _ids_input(rng, dims, seq="enc")
    dec_nt, dec_ids = _ids_input(rng, dims, seq="dec")
    data["encoder_input_ids"] = enc_nt
    data["decoder_input_ids"] = dec_nt
    data["decoder_attention_mask"] = _sdpa_causal_mask_fortran(dims.decoder_seq)
    data["cross_attention_mask"] = _cross_attn_mask_fortran(
        dims.encoder_seq, dims.decoder_seq,
    )
    dec_m = torch.tensor(
        data["decoder_attention_mask"].reshape(
            dims.decoder_seq, dims.decoder_seq,
        ),
    )
    cross_m = torch.tensor(
        data["cross_attention_mask"].reshape(
            dims.encoder_seq, dims.decoder_seq,
        ),
    )
    vocab = _weight_from_fortran(
        data["conditional.model.embed_tokens.vocab"],
    ).T.requires_grad_(True)
    hidden_dsb = _pt_t5_model(
        enc_ids, dec_ids, data, "conditional.model", dims, dec_m, cross_m,
        vocab=vocab,
    )
    hidden_bsd = hidden_dsb.permute(2, 1, 0).contiguous()
    lm_w = _weight_from_fortran(data["conditional.lm_head.weight"])
    lm_w = lm_w.requires_grad_(True)
    logits = hidden_bsd @ lm_w
    out = logits.permute(0, 2, 1).contiguous()
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_embed_tokens_vocab"] = fortran_order(
        vocab.grad.detach().numpy().T,
    )
    return data


GENERATORS = {
    "ff": generate_ff,
    "attention": lambda seed: generate_attention(seed, causal=False),
    "attention_causal": lambda seed: generate_attention(seed, causal=True),
    "cross_attention": generate_cross_attention,
    "encoder_block": generate_encoder_block,
    "decoder_block": generate_decoder_block,
    "model": generate_model,
    "conditional": generate_conditional,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate T5 block test data")
    parser.add_argument("--block", choices=GENERATORS, required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--seed", "-s", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    data = GENERATORS[args.block](args.seed)
    stem = f"t5_{args.block}"
    bundle_path = str(out / f"{stem}.safetensors")
    save_file(data, bundle_path)
    print(f"Saved {bundle_path}")

    tol = 2e-5
    if args.block == "ff":
        write_fixture_json(out, stem, FF_DIMS, tol, tol)
    elif args.block in ("attention", "attention_causal"):
        write_fixture_json(out, stem, ATTN_DIMS, tol, tol)
    elif args.block == "cross_attention":
        write_fixture_json(out, stem, CROSS_DIMS, tol, tol, enc_seq=CROSS_DIMS.encoder_seq, dec_seq=CROSS_DIMS.decoder_seq)
    elif args.block == "encoder_block":
        write_fixture_json(out, stem, ENCODER_BLOCK_DIMS, tol, tol)
    elif args.block == "decoder_block":
        write_fixture_json(out, stem, DECODER_BLOCK_DIMS, tol, tol, enc_seq=DECODER_BLOCK_DIMS.encoder_seq, dec_seq=DECODER_BLOCK_DIMS.decoder_seq)
    elif args.block in ("model", "conditional"):
        write_fixture_json(out, stem, MODEL_DIMS, tol, tol, enc_seq=MODEL_DIMS.encoder_seq, dec_seq=MODEL_DIMS.decoder_seq)

    return 0


if __name__ == "__main__":
    sys.exit(main())
