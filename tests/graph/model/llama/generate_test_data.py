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

For each block the script creates a ``llama_<block>_full.safetensors`` file
that stores NNTile-layout weights, input tensor(s), the PyTorch reference
forward output, and backward reference gradients (upstream gradient,
input gradient).  Both NNTile and PyTorch are configured identically:

  * RoPE is replaced with identity (cos=1, sin=0).
  * Causal attention mask is disabled (full attention).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention as PtAttention, LlamaDecoderLayer as PtDecoderLayer,
    LlamaForCausalLM as PtCausalLM, LlamaMLP as PtMLP, LlamaModel as PtModel)


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


MHA_DIMS = TestDims(
    hidden=8, intermediate=16, n_heads=1, kv_heads=1,
    seq=4, batch=2, vocab=100, num_layers=2,
)

GQA_DIMS = TestDims(
    hidden=8, intermediate=16, n_heads=4, kv_heads=2,
    seq=4, batch=2, vocab=100, num_layers=2,
)


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
        _attn_implementation="eager",
        base_model_tp_plan=None,
    )


def _identity_pos_emb(dims: TestDims):
    """Return ``(cos, sin)`` that make RoPE an identity transform."""
    cos = torch.ones(dims.batch, dims.seq, dims.head_size)
    sin = torch.zeros_like(cos)
    return cos, sin


class _IdentityRoPE(torch.nn.Module):
    """Drop-in replacement for ``LlamaRotaryEmbedding`` (identity)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x, position_ids):
        bs, seq_len = x.shape[0], x.shape[1]
        cos = torch.ones(bs, seq_len, self.dim, device=x.device, dtype=x.dtype)
        sin = torch.zeros_like(cos)
        return cos, sin


def _patch_model(model, dims: TestDims):
    """Disable RoPE and causal mask for NNTile (no-RoPE, no-mask) match."""
    target = getattr(model, "model", model)
    if hasattr(target, "rotary_emb"):
        target.rotary_emb = _IdentityRoPE(dims.head_size)
    if hasattr(target, "_update_causal_mask"):
        target._update_causal_mask = lambda *a, **kw: None


# ── Weight-extraction (PyTorch → NNTile layout) ─────────────────────────


def _linear(linear: torch.nn.Linear) -> np.ndarray:
    """PT Linear weight ``(out, in)`` C → NNTile ``(in, out)`` Fortran."""
    return fortran_order(linear.weight.detach().numpy().T)


def _attn(attn, prefix: str, dims: TestDims) -> dict[str, np.ndarray]:
    q = attn.q_proj.weight.detach().numpy()
    k = attn.k_proj.weight.detach().numpy()
    v = attn.v_proj.weight.detach().numpy()
    o = attn.o_proj.weight.detach().numpy()
    n_emb = q.shape[1]

    if dims.use_gqa:
        # Reshape then transpose so that NNTile's (g, h) maps to
        # HF head h*kv_group_size+g, matching the KV-group broadcast.
        q_arr = q.reshape(
            dims.kv_heads, dims.kv_group_size, dims.head_size, n_emb,
        ).transpose(1, 0, 2, 3)
        o_arr = o.reshape(
            n_emb, dims.kv_heads, dims.kv_group_size, dims.head_size,
        ).transpose(0, 2, 1, 3)
    else:
        q_arr = q.reshape(dims.n_heads, dims.head_size, n_emb)
        o_arr = o.reshape(n_emb, dims.n_heads, dims.head_size)

    return {
        f"{prefix}.q_weight": fortran_order(q_arr),
        f"{prefix}.k_weight": fortran_order(
            k.reshape(dims.kv_heads, dims.head_size, n_emb)
        ),
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


def _decoder_layer(layer, prefix: str, dims: TestDims) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {}
    d.update(_rms(layer.input_layernorm, f"{prefix}.input_norm"))
    d.update(_attn(layer.self_attn, f"{prefix}.attention", dims))
    d.update(
        _rms(layer.post_attention_layernorm, f"{prefix}.post_attn_norm")
    )
    d.update(_mlp(layer.mlp, f"{prefix}.mlp"))
    return d


def _model_weights(model, prefix: str, dims: TestDims) -> dict[str, np.ndarray]:
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
    ids = rng.integers(0, dims.vocab, size=(dims.seq, dims.batch)).astype(np.int64)
    ids_nt = ids.ravel("F").reshape(ids.shape)
    ids_pt = torch.tensor(ids.T.copy(), dtype=torch.long)
    return ids_nt, ids_pt


def _out_to_nntile(pt_out: torch.Tensor) -> np.ndarray:
    """PT output ``(B, S, D)`` → NNTile ``(D, S, B)`` Fortran."""
    return fortran_order(pt_out.detach().numpy().transpose(2, 1, 0))


# ── Block generators ─────────────────────────────────────────────────────


def generate_attention(seed: int, dims: TestDims = MHA_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)

    pt = PtAttention(config, layer_idx=0)
    pt.eval()

    data = _attn(pt, "attn", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt

    pos = _identity_pos_emb(dims)
    out = pt(x_pt, position_embeddings=pos, attention_mask=None)[0]
    data["output_ref"] = _out_to_nntile(out)

    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data


def generate_mlp(seed: int, dims: TestDims = MHA_DIMS) -> dict[str, np.ndarray]:
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


def generate_decoder(seed: int, dims: TestDims = MHA_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)

    pt = PtDecoderLayer(config, layer_idx=0)
    pt.eval()

    data = _decoder_layer(pt, "decoder", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt

    pos = _identity_pos_emb(dims)
    out = pt(x_pt, position_embeddings=pos)[0]
    data["output_ref"] = _out_to_nntile(out)

    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data


def generate_model(seed: int, dims: TestDims = MHA_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)

    pt = PtModel(config)
    _patch_model(pt, dims)

    data = _model_weights(pt, "model", dims)
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt

    out = pt(ids_pt).last_hidden_state
    data["output_ref"] = _out_to_nntile(out)

    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    return data


def generate_causal(seed: int, dims: TestDims = MHA_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)

    pt = PtCausalLM(config)
    _patch_model(pt, dims)

    data = _model_weights(pt.model, "model.model", dims)
    data["model.lm_head.weight"] = _linear(pt.lm_head)
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt

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
    "attention_gqa": lambda seed: generate_attention(seed, GQA_DIMS),
    "decoder_gqa": lambda seed: generate_decoder(seed, GQA_DIMS),
    "model_gqa": lambda seed: generate_model(seed, GQA_DIMS),
    "causal_gqa": lambda seed: generate_causal(seed, GQA_DIMS),
}

# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Llama block test data (safetensors)",
    )
    parser.add_argument(
        "--block",
        required=True,
        choices=GENERATORS,
        help="Llama block to generate data for",
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

    full_path = str(out / f"llama_{args.block}_full.safetensors")
    save_file(data, full_path)
    print(f"Saved {full_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
