#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file nntile/tests/model/gpt2/generate_test_data.py
# Generate GPT-2 building-block test data in safetensors format.
#
# @version 1.1.0

"""Generate reference test data for NNTile GPT-2 graph C++ tests.

For each block the script creates ``gpt2_<block>.safetensors`` plus a paired
``.json`` sidecar (geometry, tolerances) read by the corresponding C++ tests.

Uses HuggingFace ``modeling_gpt2`` for all forward/backward references
(``GPT2MLP``, ``GPT2Attention``, ``GPT2Block``, ``GPT2Model``,
``GPT2LMHeadModel``) plus NumPy layout helpers. Weight tensors are reshaped to
the graph module layout; reference forwards call HF modules (or
``eager_attention_forward`` from the same file for bidirectional attention
without a causal mask).
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
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2MLP as PtMLP, GPT2Attention as PtAttention, GPT2Block as PtBlock,
    GPT2LMHeadModel as PtCausalLM, GPT2Model as PtModel,
    eager_attention_forward)

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

BLOCK_DIMS = ATTENTION_DIMS
MODEL_DIMS = ATTENTION_DIMS
CAUSAL_DIMS = ATTENTION_DIMS


def fortran_order(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    return a.ravel("F").reshape(a.shape)


def fortran_order_int64(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.int64)
    return a.ravel("F").reshape(a.shape)


def _make_config(dims: TestDims) -> GPT2Config:
    return GPT2Config(
        n_embd=dims.hidden,
        n_inner=dims.intermediate,
        n_head=dims.n_heads,
        n_layer=dims.num_layers,
        vocab_size=dims.vocab,
        n_positions=max(dims.seq * 2, 128),
        layer_norm_epsilon=dims.layer_norm_eps,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        _attn_implementation="eager",
    )


def _lm_head_to_linear_weight(conv) -> np.ndarray:
    """``lm_head`` Conv1D ``(vocab, hidden)`` → Linear ``(hidden, vocab)``."""
    return fortran_order(conv.weight.detach().numpy().T)


def _conv1d_to_linear_weight(conv) -> np.ndarray:
    """HF Conv1D ``(in, out)`` → graph Linear (same shape)."""
    return fortran_order(conv.weight.detach().numpy())


def _layer_norm(ln, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.gamma": fortran_order(ln.weight.detach().numpy()),
        f"{prefix}.beta": fortran_order(ln.bias.detach().numpy()),
    }


def _gpt2_attn_weights(
    attn: PtAttention, prefix: str, dims: TestDims,
) -> dict[str, np.ndarray]:
    """Split HF ``c_attn`` into graph ``q/k/v`` layouts; ``c_proj`` → ``o``.

    Matches ``GPT2Attention.from_torch`` in
    ``wrappers/python/nntile/model/gpt2_attention.py``.
    """
    w = attn.c_attn.weight.detach().numpy()
    n_emb = dims.hidden
    hs = dims.head_size
    n_heads = dims.n_heads
    q_arr = w[:, 0:n_emb].T.reshape(n_heads, hs, n_emb)
    k_arr = w[:, n_emb:2 * n_emb].T.reshape(n_heads, hs, n_emb)
    v_arr = w[:, 2 * n_emb:3 * n_emb].T.reshape(n_heads, hs, n_emb)
    o_arr = attn.c_proj.weight.detach().numpy().T.reshape(
        n_emb, n_heads, hs,
    )
    bias = attn.c_attn.bias.detach().numpy()
    b_q = bias[0:n_emb].reshape(n_heads, hs).T
    b_k = bias[n_emb:2 * n_emb].reshape(n_heads, hs).T
    b_v = bias[2 * n_emb:3 * n_emb].reshape(n_heads, hs).T
    return {
        f"{prefix}.q_weight": fortran_order(q_arr),
        f"{prefix}.k_weight": fortran_order(k_arr),
        f"{prefix}.v_weight": fortran_order(v_arr),
        f"{prefix}.o_weight": fortran_order(o_arr),
        f"{prefix}.q_bias": fortran_order(b_q),
        f"{prefix}.k_bias": fortran_order(b_k),
        f"{prefix}.v_bias": fortran_order(b_v),
        f"{prefix}.o_bias": fortran_order(
            attn.c_proj.bias.detach().numpy()),
    }


def _gpt2_mlp(mlp: PtMLP, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.fc1.weight": _conv1d_to_linear_weight(mlp.c_fc),
        f"{prefix}.fc1.bias": fortran_order(mlp.c_fc.bias.detach().numpy()),
        f"{prefix}.fc2.weight": _conv1d_to_linear_weight(mlp.c_proj),
        f"{prefix}.fc2.bias": fortran_order(mlp.c_proj.bias.detach().numpy()),
    }


def _gpt2_block(
    layer: PtBlock, prefix: str, dims: TestDims,
) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {}
    d.update(_layer_norm(layer.ln_1, f"{prefix}.ln_1"))
    d.update(_gpt2_attn_weights(layer.attn, f"{prefix}.attn", dims))
    d.update(_layer_norm(layer.ln_2, f"{prefix}.ln_2"))
    d.update(_gpt2_mlp(layer.mlp, f"{prefix}.mlp"))
    return d


def _embed(embed, prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}.vocab": fortran_order(embed.weight.detach().numpy().T)}


def _model_weights(
    model: PtModel, prefix: str, dims: TestDims,
) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {}
    d.update(_embed(model.wte, f"{prefix}.wte"))
    d.update(_embed(model.wpe, f"{prefix}.wpe"))
    d.update(_layer_norm(model.ln_f, f"{prefix}.ln_f"))
    for i, layer in enumerate(model.h):
        d.update(_gpt2_block(layer, f"{prefix}.h_{i}", dims))
    return d


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


def _position_ids(dims: TestDims) -> np.ndarray:
    pos = np.arange(dims.seq, dtype=np.int64)[:, None]
    pos = np.broadcast_to(pos, (dims.seq, dims.batch)).copy()
    return fortran_order_int64(pos)


def _out_to_nntile(pt_out: torch.Tensor) -> np.ndarray:
    return fortran_order(pt_out.detach().numpy().transpose(2, 1, 0))


def _sdpa_causal_mask_fortran(seq: int) -> np.ndarray:
    kk = np.arange(seq, dtype=np.int64)[:, None]
    qq = np.arange(seq, dtype=np.int64)[None, :]
    allowed = (kk <= qq).astype(np.float32)
    return fortran_order(allowed)


def _causal_additive_mask_torch(
    batch: int, seq: int, device: torch.device,
) -> torch.Tensor:
    mask = np.array(np.triu(np.ones((seq, seq))), dtype=bool, order="F")
    mask_torch = torch.tensor(
        np.array(1 - mask, dtype=np.float32),
    ).T * torch.finfo(torch.float32).min
    mask_torch = mask_torch.to(device=device, dtype=torch.float32)
    return mask_torch[None, None, :, :].expand(batch, 1, -1, -1)


def _gpt2_fixture_json(
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
        "gpt2": {
            "vocab_size": dims.vocab,
            "hidden_size": dims.hidden,
            "intermediate_size": dims.intermediate,
            "num_hidden_layers": dims.num_layers,
            "num_attention_heads": dims.n_heads,
            "max_position_embeddings": max(dims.seq * 2, 128),
            "layer_norm_eps": dims.layer_norm_eps,
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
            _gpt2_fixture_json(stem, dims, forward_tol, backward_tol),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {path}")


def generate_mlp(
    seed: int, dims: TestDims = MLP_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtMLP(dims.intermediate, config)
    pt.eval()
    data = _gpt2_mlp(pt, "mlp")
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    out = pt(x_pt)
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data


def _gpt2_attn_forward_hf(
    attn: PtAttention,
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    bidirectional: bool = False,
) -> torch.Tensor:
    """HF GPT-2 attention forward (``_attn_implementation="eager"``).

    Causal refs use the built-in lower-triangular mask
    (``bidirectional=False``). For graph ``mask=nullptr`` tests,
    ``bidirectional=True`` temporarily sets ``is_cross_attention`` so
    ``eager_attention_forward`` skips the causal mask
    (``GPT2Attention.forward`` always applies causal masking in eager mode
    when ``attention_mask`` is None).
    """
    query_states, key_states, value_states = attn.c_attn(hidden_states).split(
        attn.split_size, dim=2,
    )
    shape_q = (*query_states.shape[:-1], -1, attn.head_dim)
    query_states = query_states.view(shape_q).transpose(1, 2)
    key_states = key_states.view(shape_q).transpose(1, 2)
    value_states = value_states.view(shape_q).transpose(1, 2)

    saved_cross = attn.is_cross_attention
    try:
        if bidirectional:
            attn.is_cross_attention = True
        attn_output, _ = eager_attention_forward(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
        )
    finally:
        attn.is_cross_attention = saved_cross

    attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
    attn_output = attn.c_proj(attn_output)
    return attn.resid_dropout(attn_output)


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
    data = _gpt2_attn_weights(pt, "attn", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    if use_causal_mask:
        attn_mask = _causal_additive_mask_torch(
            dims.batch, dims.seq, x_pt.device,
        )
        data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
        out = _gpt2_attn_forward_hf(
            pt, x_pt, attention_mask=attn_mask, bidirectional=False,
        )
    else:
        out = _gpt2_attn_forward_hf(pt, x_pt, bidirectional=True)
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data


def generate_block(
    seed: int, dims: TestDims = BLOCK_DIMS, *, use_causal_mask: bool = True,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtBlock(config, layer_idx=0)
    pt.eval()
    data = _gpt2_block(pt, "block", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    attn_mask = _causal_additive_mask_torch(
        dims.batch, dims.seq, x_pt.device,
    )
    data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
    out = pt(x_pt, attention_mask=attn_mask)[0]
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
    data["position_ids"] = _position_ids(dims)
    data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
    attn_mask = _causal_additive_mask_torch(
        dims.batch, dims.seq, ids_pt.device,
    )
    out = pt(
        input_ids=ids_pt,
        attention_mask=attn_mask,
    ).last_hidden_state
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    out.backward(g_pt)
    data["grad_output"] = g_nt
    data["grad_wte_vocab"] = fortran_order(
        pt.wte.weight.grad.detach().numpy().T)
    return data


def generate_causal(
    seed: int, dims: TestDims = CAUSAL_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtCausalLM(config)
    pt.eval()
    data = _model_weights(pt.transformer, "model.transformer", dims)
    data["model.lm_head.weight"] = _lm_head_to_linear_weight(pt.lm_head)
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt
    data["position_ids"] = _position_ids(dims)
    data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
    attn_mask = _causal_additive_mask_torch(
        dims.batch, dims.seq, ids_pt.device,
    )
    out = pt(input_ids=ids_pt, attention_mask=attn_mask).logits
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    out.backward(g_pt)
    data["grad_output"] = g_nt
    data["grad_wte_vocab"] = fortran_order(
        pt.transformer.wte.weight.grad.detach().numpy().T)
    return data


GENERATORS = {
    "mlp": generate_mlp,
    "attention": lambda seed: generate_attention(seed, use_causal_mask=False),
    "attention_causal": lambda seed: generate_attention(
        seed, use_causal_mask=True,
    ),
    "block": generate_block,
    "model": generate_model,
    "causal": generate_causal,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate GPT-2 block test data (safetensors)",
    )
    parser.add_argument(
        "--block",
        choices=GENERATORS,
        required=True,
        help="GPT-2 block to generate data for",
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
    stem = f"gpt2_{args.block}"
    bundle_path = str(out / f"{stem}.safetensors")
    save_file(data, bundle_path)
    print(f"Saved {bundle_path}")

    tol = 1e-6
    if args.block == "mlp":
        write_fixture_json(out, stem, MLP_DIMS, tol, tol)
    elif args.block in ("attention", "attention_causal"):
        write_fixture_json(out, stem, ATTENTION_DIMS, tol, tol)
    elif args.block == "block":
        write_fixture_json(out, stem, BLOCK_DIMS, tol, tol)
    elif args.block in ("model", "causal"):
        write_fixture_json(out, stem, MODEL_DIMS, tol, tol)

    return 0


if __name__ == "__main__":
    sys.exit(main())
