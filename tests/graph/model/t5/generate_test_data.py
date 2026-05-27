#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file tests/graph/model/t5/generate_test_data.py
# Generate T5 building-block test data in safetensors format.
#
# @version 1.2.0

"""Generate reference test data for NNTile T5 graph C++ tests.

Uses **Hugging Face Transformers** (``modeling_t5``) for forward and backward
references. Weight tensors are converted to the NNTile graph layout (same
naming and Fortran-order bytes as ``examples/t5_generate.py``). Mask tensors
stored for C++ use the ``sdpa_eager`` layout expected by graph tests.

PyTorch runs with ``_attn_implementation="eager"`` and ``cache_position`` on
attention modules, matching
``wrappers/python/tests/layer/test_t5_attention.py``.
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
from transformers import T5Config
from transformers.models.t5.modeling_t5 import (
    T5Attention as PtAttention, T5Block as PtBlock,
    T5ForConditionalGeneration as PtConditional, T5LayerFF as PtLayerFF,
    T5Model as PtModel)

# Graph ``T5Attention`` has no learned relative-position bias (T5 RoPE/RPE).
# HF enables it on the first layer of each stack; turn it off for references.
def _disable_self_attn_relative_bias(block: PtBlock) -> None:
    block.layer[0].SelfAttention.has_relative_attention_bias = False

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
    """C-contiguous flat bytes matching NNTile Fortran tile linearization."""
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


def _rms_gamma(
    layer_norm: torch.nn.Module, prefix: str,
) -> dict[str, np.ndarray]:
    w = layer_norm.weight.detach().numpy()
    return {f"{prefix}.gamma": fortran_order(w)}


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


def _encoder_block_weights(
    block: PtBlock, prefix: str, dims: TestDims,
) -> dict:
    d: dict[str, np.ndarray] = {}
    d.update(_rms_gamma(block.layer[0].layer_norm, f"{prefix}.layer_norm_0"))
    d.update(
        _t5_attn_weights(
            block.layer[0].SelfAttention, f"{prefix}.self_attn", dims,
        ),
    )
    d.update(_t5_ff_weights(block.layer[1], f"{prefix}.ff"))
    return d


def _decoder_block_weights(
    block: PtBlock, prefix: str, dims: TestDims,
) -> dict:
    d: dict[str, np.ndarray] = {}
    d.update(_rms_gamma(block.layer[0].layer_norm, f"{prefix}.layer_norm_0"))
    d.update(
        _t5_attn_weights(
            block.layer[0].SelfAttention, f"{prefix}.self_attn", dims,
        ),
    )
    d.update(_rms_gamma(block.layer[1].layer_norm, f"{prefix}.layer_norm_1"))
    d.update(
        _t5_attn_weights(
            block.layer[1].EncDecAttention, f"{prefix}.cross_attn", dims,
        ),
    )
    d.update(_t5_ff_weights(block.layer[2], f"{prefix}.ff"))
    return d


def _embed_weights(
    embed: torch.nn.Embedding, prefix: str,
) -> dict[str, np.ndarray]:
    w = embed.weight.detach().numpy().T
    return {f"{prefix}.vocab": fortran_order(w)}


def _model_weights(model: PtModel, prefix: str, dims: TestDims) -> dict:
    d: dict[str, np.ndarray] = {}
    d.update(_embed_weights(model.shared, f"{prefix}.embed_tokens"))
    d.update(
        _rms_gamma(
            model.encoder.final_layer_norm, f"{prefix}.encoder_final_norm",
        ),
    )
    d.update(
        _rms_gamma(
            model.decoder.final_layer_norm, f"{prefix}.decoder_final_norm",
        ),
    )
    for i, layer in enumerate(model.encoder.block):
        p = f"{prefix}.encoder_layers_{i}"
        d.update(_encoder_block_weights(layer, p, dims))
    for i, layer in enumerate(model.decoder.block):
        p = f"{prefix}.decoder_layers_{i}"
        d.update(_decoder_block_weights(layer, p, dims))
    return d


def _conditional_weights(
    model: PtConditional, prefix: str, dims: TestDims,
) -> dict:
    d = _model_weights(model, f"{prefix}.model", dims)
    d[f"{prefix}.lm_head.weight"] = fortran_order(
        model.lm_head.weight.detach().numpy().T,
    )
    return d


# ── Input / output / mask helpers ─────────────────────────────────────────


def _hidden_input(
    rng, dims: TestDims, *, seq: int | None = None, scale: float = 0.1,
):
    s = dims.seq if seq is None else seq
    x = rng.standard_normal((dims.d_model, s, dims.batch))
    x = x.astype(np.float32) * scale
    x_nt = fortran_order(x)
    x_pt = torch.tensor(x.transpose(2, 1, 0).copy(), requires_grad=True)
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


def _cache_position(hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.arange(
        hidden_states.shape[1],
        dtype=torch.long,
        device=hidden_states.device,
    )


def _hf_causal_attention_mask_4d(
    batch: int,
    seq: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive causal mask ``(batch, 1, seq, seq)`` for HF ``T5Attention``."""
    mask = torch.zeros(batch, 1, seq, seq, device=device, dtype=dtype)
    upper = torch.triu(
        torch.ones(seq, seq, device=device, dtype=torch.bool), diagonal=1,
    )
    min_val = torch.finfo(dtype).min
    return mask.masked_fill(upper.unsqueeze(0).unsqueeze(0), min_val)


def _sdpa_causal_mask_fortran(seq: int) -> np.ndarray:
    kk = np.arange(seq, dtype=np.int64)[:, None]
    qq = np.arange(seq, dtype=np.int64)[None, :]
    return fortran_order((kk <= qq).astype(np.float32))


def _cross_attn_mask_fortran(enc_seq: int, dec_seq: int) -> np.ndarray:
    """``(k_seq, q_seq)`` = ``(enc_seq, dec_seq)`` for graph ``sdpa_eager``."""
    return fortran_order(np.ones((enc_seq, dec_seq), dtype=np.float32))


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
    out: Path,
    stem: str,
    dims: TestDims,
    forward_tol: float,
    backward_tol: float,
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


# ── Block generators (Hugging Face forward / backward) ────────────────────


def generate_ff(seed: int, dims: TestDims = FF_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtLayerFF(config)
    pt.eval()
    data = _t5_ff_weights(pt, "ff")
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    _run_fwd_bwd(lambda x: pt(x), x_pt, rng, data)
    return data


def write_attention_no_rope_mask_variant_files(out: Path, seed: int) -> None:
    """Write self-attention fixtures for the no-RoPE / mask matrix.

    T5 has no RoPE. Measured C++ vs Hugging Face reference (seed 42,
    ``ATTN_DIMS``) after graph SDPA scale=1:
    - ``t5_attention_no_rope_nomask``: forward ~1e-6, backward ~1e-6
    - ``t5_attention_no_rope_causal``: forward ~5e-8, backward ~2e-7
    """
    specs: list[tuple[str, bool, float, float]] = [
        ("t5_attention_no_rope_nomask", False, 3e-7, 5e-7),
        ("t5_attention_no_rope_causal", True, 3e-7, 5e-7),
    ]
    for stem, causal, fwd_tol, bwd_tol in specs:
        payload = generate_attention(seed, ATTN_DIMS, causal=causal)
        path = str(out / f"{stem}.safetensors")
        save_file(payload, path)
        print(f"Saved {path}")
        write_fixture_json(out, stem, ATTN_DIMS, fwd_tol, bwd_tol)


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
    cp = _cache_position(x_pt)
    mask_4d = None
    if causal:
        data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
        mask_4d = _hf_causal_attention_mask_4d(
            dims.batch, dims.seq, device=x_pt.device, dtype=x_pt.dtype,
        )

    def fwd(x: torch.Tensor) -> torch.Tensor:
        out, _, _ = pt(x, mask=mask_4d, cache_position=cp)
        return out

    _run_fwd_bwd(fwd, x_pt, rng, data)
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
    enc_pt = enc_pt.detach()
    data["input"] = x_nt
    data["encoder_input"] = enc_nt
    data["cross_attn_mask"] = _cross_attn_mask_fortran(
        dims.encoder_seq, dims.decoder_seq,
    )
    cp = _cache_position(x_pt)

    def fwd(x: torch.Tensor) -> torch.Tensor:
        out, _, _ = pt(
            x, key_value_states=enc_pt, cache_position=cp,
        )
        return out

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
    _disable_self_attn_relative_bias(block)
    block.eval()
    data = _encoder_block_weights(block, "encoder", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    cp = _cache_position(x_pt)

    def fwd(x: torch.Tensor) -> torch.Tensor:
        return block(x, cache_position=cp)[0]

    _run_fwd_bwd(fwd, x_pt, rng, data)
    return data


def generate_decoder_block(
    seed: int, dims: TestDims = DECODER_BLOCK_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt_model = PtModel(config)
    block = pt_model.decoder.block[0]
    _disable_self_attn_relative_bias(block)
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
    cp = _cache_position(x_pt)
    dec_mask = _hf_causal_attention_mask_4d(
        dims.batch, dims.decoder_seq, device=x_pt.device, dtype=x_pt.dtype,
    )

    def fwd(x: torch.Tensor) -> torch.Tensor:
        return block(
            x,
            attention_mask=dec_mask,
            encoder_hidden_states=enc_pt,
            cache_position=cp,
        )[0]

    _run_fwd_bwd(fwd, x_pt, rng, data)
    return data


def generate_model(
    seed: int, dims: TestDims = MODEL_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtModel(config)
    for enc_layer in pt.encoder.block:
        _disable_self_attn_relative_bias(enc_layer)
    for dec_layer in pt.decoder.block:
        _disable_self_attn_relative_bias(dec_layer)
    pt.eval()
    data = _model_weights(pt, "model", dims)
    enc_nt, enc_ids = _ids_input(rng, dims, seq="enc")
    dec_nt, dec_ids = _ids_input(rng, dims, seq="dec")
    data["encoder_input_ids"] = enc_nt
    data["decoder_input_ids"] = dec_nt
    data["decoder_attention_mask"] = _sdpa_causal_mask_fortran(
        dims.decoder_seq,
    )
    data["cross_attention_mask"] = _cross_attn_mask_fortran(
        dims.encoder_seq, dims.decoder_seq,
    )
    pt.shared.weight.requires_grad_(True)
    out = pt(
        input_ids=enc_ids,
        decoder_input_ids=dec_ids,
    ).last_hidden_state
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_embed_tokens_vocab"] = fortran_order(
        pt.shared.weight.grad.detach().numpy().T,
    )
    return data


def generate_conditional(
    seed: int, dims: TestDims = CONDITIONAL_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtConditional(config)
    for enc_layer in pt.encoder.block:
        _disable_self_attn_relative_bias(enc_layer)
    for dec_layer in pt.decoder.block:
        _disable_self_attn_relative_bias(dec_layer)
    pt.eval()
    data = _conditional_weights(pt, "conditional", dims)
    enc_nt, enc_ids = _ids_input(rng, dims, seq="enc")
    dec_nt, dec_ids = _ids_input(rng, dims, seq="dec")
    data["encoder_input_ids"] = enc_nt
    data["decoder_input_ids"] = dec_nt
    data["decoder_attention_mask"] = _sdpa_causal_mask_fortran(
        dims.decoder_seq,
    )
    data["cross_attention_mask"] = _cross_attn_mask_fortran(
        dims.encoder_seq, dims.decoder_seq,
    )
    vocab = pt.shared.weight.detach().clone().requires_grad_(True)
    lm_w = pt.lm_head.weight.detach()

    def _embed_hook(_module, inp, _out):
        return torch.nn.functional.embedding(inp[0], vocab)

    def _lm_head_hook(_module, inp, _out):
        return inp[0] @ lm_w.T

    embed_hook = pt.shared.register_forward_hook(_embed_hook)
    lm_hook = pt.lm_head.register_forward_hook(_lm_head_hook)
    try:
        logits = pt(
            input_ids=enc_ids,
            decoder_input_ids=dec_ids,
        ).logits
        data["output_ref"] = _out_to_nntile(logits)
        g_nt, g_pt = _grad_output(rng, logits)
        data["grad_output"] = g_nt
        logits.backward(g_pt)
        data["grad_embed_tokens_vocab"] = fortran_order(
            vocab.grad.detach().numpy().T,
        )
    finally:
        embed_hook.remove()
        lm_hook.remove()
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

# Per-block Frobenius tolerances (C++ graph vs HF reference, seed 42).
# FF uses GELUTANH (HF ``gelu_new``); measured ~2e-7 forward, ~4e-7 backward.
# Attention: measured ~1e-7 forward (causal ~5e-8), ~2e-7 backward.
# Stacked blocks: measured ~1.2e-5 forward, ~1.6e-5 backward (mask mismatch).
# Conditional forward: measured ~3e-7 after ``d_model**-0.5`` prescale.
BLOCK_TOLERANCES: dict[str, tuple[float, float]] = {
    "ff": (5e-7, 1e-6),
    "attention": (3e-7, 5e-7),
    "attention_causal": (3e-7, 5e-7),
    "cross_attention": (3e-7, 5e-7),
    "encoder_block": (1.5e-5, 2e-5),
    "decoder_block": (1.5e-5, 2e-5),
    "model": (1.5e-5, 2e-5),
    "conditional": (1e-6, 2e-5),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate T5 block test data")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--block", choices=GENERATORS)
    mode.add_argument(
        "--write-attention-no-rope-mask-variants",
        action="store_true",
        help=(
            "Write ``t5_attention_no_rope_nomask`` and "
            "``t5_attention_no_rope_causal`` for C++ graph tests."
        ),
    )
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--seed", "-s", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.write_attention_no_rope_mask_variants:
        write_attention_no_rope_mask_variant_files(out, args.seed)
        return 0

    data = GENERATORS[args.block](args.seed)
    stem = f"t5_{args.block}"
    bundle_path = str(out / f"{stem}.safetensors")
    save_file(data, bundle_path)
    print(f"Saved {bundle_path}")

    fwd_tol, bwd_tol = BLOCK_TOLERANCES[args.block]
    if args.block == "ff":
        write_fixture_json(out, stem, FF_DIMS, fwd_tol, bwd_tol)
    elif args.block in ("attention", "attention_causal"):
        write_fixture_json(out, stem, ATTN_DIMS, fwd_tol, bwd_tol)
    elif args.block == "cross_attention":
        write_fixture_json(
            out, stem, CROSS_DIMS, fwd_tol, bwd_tol,
            enc_seq=CROSS_DIMS.encoder_seq, dec_seq=CROSS_DIMS.decoder_seq,
        )
    elif args.block == "encoder_block":
        write_fixture_json(out, stem, ENCODER_BLOCK_DIMS, fwd_tol, bwd_tol)
    elif args.block == "decoder_block":
        write_fixture_json(
            out, stem, DECODER_BLOCK_DIMS, fwd_tol, bwd_tol,
            enc_seq=DECODER_BLOCK_DIMS.encoder_seq,
            dec_seq=DECODER_BLOCK_DIMS.decoder_seq,
        )
    elif args.block in ("model", "conditional"):
        write_fixture_json(
            out, stem, MODEL_DIMS, fwd_tol, bwd_tol,
            enc_seq=MODEL_DIMS.encoder_seq, dec_seq=MODEL_DIMS.decoder_seq,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
