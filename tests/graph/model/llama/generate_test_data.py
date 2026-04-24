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
that stores NNTile-layout weights, input tensor(s), reference forward output,
and backward reference gradients.

Uses **HuggingFace Transformers** (``modeling_llama``) plus NumPy layout
wrangling only — no NNTile Python runtime or StarPU.

``mlp`` / ``decoder`` / ``model`` / ``causal`` (and ``decoder_gqa`` /
``model_gqa`` / ``causal_gqa``) use ``MHA_DIMS`` (hidden=8) or ``GQA_DIMS``
(8-wide GQA). ``attention`` / ``attention_gqa`` use ``ATTENTION_MHA_DIMS`` /
``ATTENTION_GQA_DIMS`` (larger, ``test_llama_attention`` ``single_tile`` + the
C++ graph attention tests).

For ``attention`` / ``attention_gqa`` blocks, ``rope_sin`` / ``rope_cos`` are
the first half-channels of ``LlamaRotaryEmbedding`` cos/sin, reshaped to
``(head_dim/2, seq, batch)`` and passed through :func:`fortran_order` so the
byte layout matches the C++ graph ``bind_data`` convention. RoPE is built like
``test_llama_attention.generate_inputs``: first argument
``v_proj.weight``, ``position_ids`` in ``(batch, seq)`` from a NumPy RNG. Forward
and backward use ``LlamaAttention`` with ``_attn_implementation="eager"`` and
the same ``(cos, sin)`` tensors.

Optional causal self-attention matches ``test_llama_attention``: additive
``attention_mask`` from the upper-triangular bool pattern; the graph tests load
``attn_mask`` as float32 ``(seq, seq)`` in Fortran layout (1 = keep logits),
converted to BOOL in C++ for ``sdpa_eager`` masking.

Extra MHA/GQA safetensors (no RoPE / causal / both) are written by
``--write-attention-rope-mask-variants`` (CTest fixture for ``llama_attention``).

The ``decoder`` and ``model`` / ``causal`` (and GQA) blocks use
``LlamaRotaryEmbedding`` the same way as full-model inference, not a no-op.
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
    LlamaAttention as PtAttention,
    LlamaDecoderLayer as PtDecoderLayer,
    LlamaForCausalLM as PtCausalLM,
    LlamaMLP as PtMLP,
    LlamaModel as PtModel,
    LlamaRotaryEmbedding,
)

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


# Small bundles for ``mlp`` / ``decoder`` / ``model`` / ``causal`` and for
# ``decoder_gqa`` / ``model_gqa`` / ``causal_gqa`` (keeps these safetensors light).
# Not used for ``attention`` or ``attention_gqa``; those use
# ``ATTENTION_MHA_DIMS`` / ``ATTENTION_GQA_DIMS`` (graph + PyTorch test).
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
    seq=64, batch=3, vocab=32000, num_layers=1,
)
ATTENTION_GQA_DIMS = TestDims(
    hidden=512, intermediate=2048, n_heads=8, kv_heads=4,
    seq=64, batch=3, vocab=32000, num_layers=1,
)

# Set to True from CLI ``--print-rope`` in :func:`main` (HuggingFace cos/sin).
_print_rope: bool = False


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
        rope_theta=1.0,
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
    """HF ``(B,S,D)`` cos/sin → NNTile-graph ``(half,S,B)`` float32 (C layout)."""
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
    return mask_torch[None, None, :, :].expand(batch, 1, -1, -1).to(dtype=dtype)


def _sdpa_causal_mask_fortran(seq: int) -> np.ndarray:
    """``(k_seq, q_seq)`` mask for graph ``sdpa_eager`` (1 = keep, 0 = mask).

    Stored as float32 because ``safetensors.numpy.save_file`` upgrades numpy
    bool to F32; the C++ test converts back to BOOL for ``mask_scalar``.
    """
    kk = np.arange(seq, dtype=np.int64)[:, None]
    qq = np.arange(seq, dtype=np.int64)[None, :]
    allowed = (kk <= qq).astype(np.float32)
    return fortran_order(allowed)


def _maybe_print_rope_hf(
    cos: torch.Tensor, sin: torch.Tensor, where: str,
) -> None:
    """If ``_print_rope`` is on, show full cos/sin from ``LlamaRotaryEmbedding``."""
    if not _print_rope:
        return
    c, s = cos.detach(), sin.detach()
    is_identity = bool(
        torch.allclose(c, torch.ones_like(c), atol=1e-5)
        and torch.allclose(s, torch.zeros_like(s), atol=1e-5)
    )
    c00 = c[0, 0, :8].to(torch.float32).cpu().numpy()
    s00 = s[0, 0, :8].to(torch.float32).cpu().numpy()
    c01 = c[0, 1, :8].to(torch.float32).cpu().numpy() if c.shape[1] > 1 else c00
    print(
        f"[{where}] RoPE: shape (B, S, D) = {tuple(cos.shape)}; "
        f"all cos=1, sin=0 (identity) would be: {is_identity}",
        file=sys.stderr,
    )
    print(
        f"[{where}]   cos[0,0,:8] = {np.array2string(c00, precision=4)}",
        file=sys.stderr,
    )
    print(
        f"[{where}]   sin[0,0,:8] = {np.array2string(s00, precision=4)}",
        file=sys.stderr,
    )
    print(
        f"[{where}]   cos[0,1,:8] = {np.array2string(c01, precision=4)}",
        file=sys.stderr,
    )


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
    RoPE) and ``rope_cos`` / ``rope_sin`` are omitted so the C++ graph skips
    RoPE (nullptr sin/cos).

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
    _maybe_print_rope_hf(cos, sin, "generate_attention / LlamaRotaryEmbedding")
    if not use_rope:
        cos = torch.ones_like(cos)
        sin = torch.zeros_like(sin)
    else:
        cos_np, sin_np = _rope_half_from_hf(cos, sin, dims)
        data["rope_cos"] = fortran_order(cos_np)
        data["rope_sin"] = fortran_order(sin_np)

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


def write_attention_rope_mask_variant_files(out: Path, seed: int) -> None:
    """Write extra attention safetensors for RoPE / causal-mask combinations."""
    specs: list[tuple[str, TestDims, bool, bool]] = [
        ("llama_attention_no_rope_full.safetensors", ATTENTION_MHA_DIMS, False, False),
        ("llama_attention_causal_full.safetensors", ATTENTION_MHA_DIMS, True, True),
        ("llama_attention_no_rope_causal_full.safetensors", ATTENTION_MHA_DIMS, False, True),
        ("llama_attention_gqa_no_rope_full.safetensors", ATTENTION_GQA_DIMS, False, False),
        ("llama_attention_gqa_causal_full.safetensors", ATTENTION_GQA_DIMS, True, True),
        ("llama_attention_gqa_no_rope_causal_full.safetensors", ATTENTION_GQA_DIMS, False, True),
    ]
    for fname, dims, rope, causal in specs:
        payload = generate_attention(
            seed, dims, use_rope=rope, use_causal_mask=causal,
        )
        path = str(out / fname)
        save_file(payload, path)
        print(f"Saved {path}")


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
    seed: int, dims: TestDims = MHA_DIMS) -> dict[str, np.ndarray]:
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
    _maybe_print_rope_hf(cos, sin, "generate_decoder / LlamaRotaryEmbedding")
    out = pt(x_pt, position_embeddings=(cos, sin))[0]
    data["output_ref"] = _out_to_nntile(out)

    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data


def generate_model(
    seed: int, dims: TestDims = MHA_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)

    pt = PtModel(config)

    data = _model_weights(pt, "model", dims)
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt

    out = pt(ids_pt).last_hidden_state
    data["output_ref"] = _out_to_nntile(out)

    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    return data


def generate_causal(
    seed: int, dims: TestDims = MHA_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)

    pt = PtCausalLM(config)

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
    "attention_gqa": lambda seed: generate_attention(seed, ATTENTION_GQA_DIMS),
    "decoder_gqa": lambda seed: generate_decoder(seed, GQA_DIMS),
    "model_gqa": lambda seed: generate_model(seed, GQA_DIMS),
    "causal_gqa": lambda seed: generate_causal(seed, GQA_DIMS),
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
            "Write six extra attention safetensors (MHA/GQA × no-RoPE / causal / "
            "both) for C++ graph tests; does not overwrite default "
            "llama_attention(_gqa)_full.safetensors."
        ),
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed",
    )
    parser.add_argument(
        "--print-rope",
        action="store_true",
        help=(
            "Print a sample of HuggingFace RoPE cos/sin (B,S,D) for attention "
            "or decoder (GQA) blocks. Identity RoPE would be all cos=1, sin=0."
        ),
    )
    args = parser.parse_args()

    global _print_rope
    _print_rope = args.print_rope

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.write_attention_rope_mask_variants:
        write_attention_rope_mask_variant_files(out, args.seed)
        return 0

    data = GENERATORS[args.block](args.seed)

    full_path = str(out / f"llama_{args.block}_full.safetensors")
    save_file(data, full_path)
    print(f"Saved {full_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
