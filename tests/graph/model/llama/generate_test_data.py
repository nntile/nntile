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
that stores NNTile-layout weights, input tensor(s) and the PyTorch reference
output.  Both NNTile and PyTorch are configured identically:

  * RoPE is replaced with identity (cos=1, sin=0).
  * Causal attention mask is disabled (full attention).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention as PtAttention, LlamaDecoderLayer as PtDecoderLayer,
    LlamaForCausalLM as PtCausalLM, LlamaMLP as PtMLP, LlamaModel as PtModel)

# ── Test dimensions (must match C++ test configs) ────────────────────────
HIDDEN = 8
INTERMEDIATE = 16
N_HEADS = 1
KV_HEADS = 1
HEAD_SIZE = HIDDEN // N_HEADS
SEQ = 4
BATCH = 2
VOCAB = 100
NUM_LAYERS = 2
RMS_EPS = 1e-6

# ── Helpers ──────────────────────────────────────────────────────────────


def fortran_order(arr: np.ndarray) -> np.ndarray:
    """Return a Fortran-contiguous float32 copy (NNTile layout)."""
    return np.asfortranarray(arr).astype(np.float32)


def _make_config() -> LlamaConfig:
    return LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=N_HEADS,
        num_key_value_heads=KV_HEADS,
        num_hidden_layers=NUM_LAYERS,
        vocab_size=VOCAB,
        rms_norm_eps=RMS_EPS,
        max_position_embeddings=2048,
        _attn_implementation="eager",
    )


def _identity_pos_emb(batch: int, seq: int, head_dim: int):
    """Return ``(cos, sin)`` that make RoPE an identity transform."""
    cos = torch.ones(batch, seq, head_dim)
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


def _patch_model(model):
    """Disable RoPE and causal mask for NNTile (no-RoPE, no-mask) match."""
    target = getattr(model, "model", model)
    if hasattr(target, "rotary_emb"):
        target.rotary_emb = _IdentityRoPE(HEAD_SIZE)
    if hasattr(target, "_update_causal_mask"):
        target._update_causal_mask = lambda *a, **kw: None


# ── Weight-extraction (PyTorch → NNTile layout) ─────────────────────────


def _linear(linear: torch.nn.Linear) -> np.ndarray:
    """PT Linear weight ``(out, in)`` C → NNTile ``(in, out)`` Fortran."""
    return fortran_order(linear.weight.detach().numpy().T)


def _attn(attn, prefix: str) -> dict[str, np.ndarray]:
    q = attn.q_proj.weight.detach().numpy()
    k = attn.k_proj.weight.detach().numpy()
    v = attn.v_proj.weight.detach().numpy()
    o = attn.o_proj.weight.detach().numpy()
    n_emb = q.shape[1]
    return {
        f"{prefix}.q_weight": fortran_order(
            q.reshape(N_HEADS, HEAD_SIZE, n_emb)
        ),
        f"{prefix}.k_weight": fortran_order(
            k.reshape(KV_HEADS, HEAD_SIZE, n_emb)
        ),
        f"{prefix}.v_weight": fortran_order(
            v.reshape(KV_HEADS, HEAD_SIZE, n_emb)
        ),
        f"{prefix}.o_weight": fortran_order(
            o.reshape(n_emb, N_HEADS, HEAD_SIZE)
        ),
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


def _decoder_layer(layer, prefix: str) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {}
    d.update(_rms(layer.input_layernorm, f"{prefix}.input_norm"))
    d.update(_attn(layer.self_attn, f"{prefix}.attention"))
    d.update(
        _rms(layer.post_attention_layernorm, f"{prefix}.post_attn_norm")
    )
    d.update(_mlp(layer.mlp, f"{prefix}.mlp"))
    return d


def _model_weights(model, prefix: str) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {}
    d.update(_embed(model.embed_tokens, f"{prefix}.embed_tokens"))
    d.update(_rms(model.norm, f"{prefix}.norm"))
    for i, layer in enumerate(model.layers):
        d.update(_decoder_layer(layer, f"{prefix}.layers_{i}"))
    return d


# ── Input / output helpers ───────────────────────────────────────────────


def _hidden_input(rng, scale: float = 0.1):
    """Random hidden-state input: NNTile (H,S,B) Fortran + PT (B,S,H)."""
    x = rng.standard_normal((HIDDEN, SEQ, BATCH)).astype(np.float32) * scale
    x_nt = fortran_order(x)
    x_pt = torch.tensor(x.transpose(2, 1, 0).copy())
    return x_nt, x_pt


def _ids_input(rng):
    """Random token-id input: NNTile ``(S,B)`` Fortran + PT ``(B,S)``."""
    ids = rng.integers(0, VOCAB, size=(SEQ, BATCH)).astype(np.int64)
    ids_nt = np.asfortranarray(ids)
    ids_pt = torch.tensor(ids.T.copy(), dtype=torch.long)
    return ids_nt, ids_pt


def _out_to_nntile(pt_out: torch.Tensor) -> np.ndarray:
    """PT output ``(B, S, D)`` → NNTile ``(D, S, B)`` Fortran."""
    return fortran_order(pt_out.detach().numpy().transpose(2, 1, 0))


# ── Block generators ─────────────────────────────────────────────────────


def generate_attention(seed: int) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config()

    pt = PtAttention(config, layer_idx=0)
    pt.eval()

    data = _attn(pt, "attn")
    x_nt, x_pt = _hidden_input(rng)
    data["input"] = x_nt

    pos = _identity_pos_emb(BATCH, SEQ, HEAD_SIZE)
    with torch.no_grad():
        out = pt(x_pt, position_embeddings=pos)[0]
    data["output_ref"] = _out_to_nntile(out)
    return data


def generate_mlp(seed: int) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config()

    pt = PtMLP(config)
    pt.eval()

    data = _mlp(pt, "mlp")
    x_nt, x_pt = _hidden_input(rng)
    data["input"] = x_nt

    with torch.no_grad():
        out = pt(x_pt)
    data["output_ref"] = _out_to_nntile(out)
    return data


def generate_decoder(seed: int) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config()

    pt = PtDecoderLayer(config, layer_idx=0)
    pt.eval()

    data = _decoder_layer(pt, "decoder")
    x_nt, x_pt = _hidden_input(rng)
    data["input"] = x_nt

    pos = _identity_pos_emb(BATCH, SEQ, HEAD_SIZE)
    with torch.no_grad():
        out = pt(x_pt, position_embeddings=pos)[0]
    data["output_ref"] = _out_to_nntile(out)
    return data


def generate_model(seed: int) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config()

    pt = PtModel(config)
    pt.eval()
    _patch_model(pt)

    data = _model_weights(pt, "model")
    ids_nt, ids_pt = _ids_input(rng)
    data["input_ids"] = ids_nt

    with torch.no_grad():
        out = pt(ids_pt).last_hidden_state
    data["output_ref"] = _out_to_nntile(out)
    return data


def generate_causal(seed: int) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config()

    pt = PtCausalLM(config)
    pt.eval()
    _patch_model(pt)

    data = _model_weights(pt.model, "model.model")
    data["model.lm_head.weight"] = _linear(pt.lm_head)
    ids_nt, ids_pt = _ids_input(rng)
    data["input_ids"] = ids_nt

    with torch.no_grad():
        out = pt(ids_pt).logits
    data["output_ref"] = _out_to_nntile(out)
    return data


GENERATORS = {
    "attention": generate_attention,
    "mlp": generate_mlp,
    "decoder": generate_decoder,
    "model": generate_model,
    "causal": generate_causal,
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
