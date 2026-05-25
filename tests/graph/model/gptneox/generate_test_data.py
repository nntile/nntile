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
    GPTNeoXAttention as PtAttention,
    GPTNeoXForCausalLM as PtCausalLM,
    GPTNeoXLayer as PtLayer,
    GPTNeoXModel as PtModel,
    GPTNeoXMLP as PtMLP,
    GPTNeoXRotaryEmbedding,
    apply_rotary_pos_emb,
)

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
        f"{prefix}.fc2.weight": fortran_order(
            mlp.dense_4h_to_h.weight.detach().numpy().T),
    }


def _gptneox_decoder_weights(
    layer: PtLayer, prefix: str, dims: TestDims,
) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {}
    d.update(_layer_norm(layer.input_layernorm, f"{prefix}.input_norm"))
    d.update(_gptneox_attn_weights(layer.attention, f"{prefix}.attention", dims))
    d.update(_layer_norm(
        layer.post_attention_layernorm, f"{prefix}.post_attn_norm"))
    d.update(_gptneox_mlp_weights(layer.mlp, f"{prefix}.mlp"))
    return d


def _embed(embed, prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}.vocab": fortran_order(embed.weight.detach().numpy().T)}


def _model_weights(model: PtModel, prefix: str, dims: TestDims) -> dict[str, np.ndarray]:
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


def _position_ids(dims: TestDims) -> np.ndarray:
    pos = np.arange(dims.seq, dtype=np.int64)[:, None]
    pos = np.broadcast_to(pos, (dims.seq, dims.batch)).copy()
    return fortran_order_int64(pos)


def _position_ids_pt(dims: TestDims, device: torch.device) -> torch.Tensor:
    return torch.arange(
        dims.seq, device=device, dtype=torch.long,
    ).unsqueeze(0).expand(dims.batch, dims.seq)


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


def _rope_half_from_hf(
    cos: torch.Tensor, sin: torch.Tensor, dims: TestDims,
) -> tuple[np.ndarray, np.ndarray]:
    """HF ``(B,S,D)`` cos/sin → NNTile-graph ``(half,S,B)`` float32."""
    half = dims.head_size // 2
    cos_half = cos[:, :, :half].to(torch.float32).detach().cpu().numpy()
    sin_half = sin[:, :, :half].to(torch.float32).detach().cpu().numpy()
    cos_np = np.transpose(cos_half, (2, 1, 0))
    sin_np = np.transpose(sin_half, (2, 1, 0))
    return cos_np, sin_np


def _rope_from_rotary(
    rotary: GPTNeoXRotaryEmbedding,
    attn: PtAttention,
    pos_ids_pt: torch.Tensor,
    x_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    H = attn.query_key_value.weight.shape[1]
    v_slice = attn.query_key_value.weight[H * 2 : H * 3, :]
    cos, sin = rotary(v_slice, pos_ids_pt)
    return cos.to(dtype=x_dtype), sin.to(dtype=x_dtype)


def _gptneox_mlp_forward(mlp: PtMLP, x_pt: torch.Tensor) -> torch.Tensor:
    """Bias-free MLP forward (matches graph ``GptneoxMlp``)."""
    h = F.linear(x_pt, mlp.dense_h_to_4h.weight, None)
    h = mlp.act(h)
    return F.linear(h, mlp.dense_4h_to_h.weight, None)


def _gptneox_attn_forward(
    attn: PtAttention,
    x_pt: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Q/K/V + RoPE + SDPA + dense, bias-free (matches graph GptneoxAttention)."""
    n_emb = x_pt.shape[-1]
    n_head = attn.config.num_attention_heads
    head_dim = attn.head_size
    qkv = F.linear(x_pt, attn.query_key_value.weight, None)
    shape = (*x_pt.shape[:2], n_head, 3 * head_dim)
    qkv = qkv.view(*shape).transpose(1, 2)
    q, k, v = qkv.chunk(3, dim=-1)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    ctx = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        is_causal=False,
    )
    ctx = ctx.transpose(1, 2).contiguous().view(*x_pt.shape)
    return F.linear(ctx, attn.dense.weight, None)


def _gptneox_decoder_forward(
    layer: PtLayer,
    x_pt: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Matches C++ ``GptneoxDecoder`` with ``use_parallel_residual=True``."""
    residual = x_pt
    x_norm = layer.input_layernorm(x_pt)
    attn_out = _gptneox_attn_forward(
        layer.attention, x_norm, cos, sin, attn_mask=attn_mask,
    )
    post_attn = residual + attn_out
    mlp_in = layer.post_attention_layernorm(residual)
    mlp_out = _gptneox_mlp_forward(layer.mlp, mlp_in)
    return post_attn + mlp_out


def _gptneox_model_forward(
    model: PtModel,
    ids_pt: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    x = model.embed_in(ids_pt)
    for layer in model.layers:
        x = _gptneox_decoder_forward(layer, x, cos, sin, attn_mask=attn_mask)
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
    out: Path, stem: str, dims: TestDims, forward_tol: float, backward_tol: float,
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
    rotary: GPTNeoXRotaryEmbedding,
    attn: PtAttention,
    pos_ids_pt: torch.Tensor,
    dims: TestDims,
    x_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = _rope_from_rotary(rotary, attn, pos_ids_pt, x_dtype)
    cos_np, sin_np = _rope_half_from_hf(cos, sin, dims)
    data["rope_cos"] = fortran_order(cos_np)
    data["rope_sin"] = fortran_order(sin_np)
    pos_nntile = pos_ids_pt.detach().cpu().numpy().T.astype(np.int64)
    data["position_ids"] = fortran_order_int64(pos_nntile)
    return cos, sin


def generate_mlp(seed: int, dims: TestDims = MLP_DIMS) -> dict[str, np.ndarray]:
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
    rotary = GPTNeoXRotaryEmbedding(config, device=torch.device("cpu"))
    data = _gptneox_attn_weights(pt, "attn", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    pos_ids_pt = _position_ids_pt(dims, x_pt.device)
    cos, sin = _write_rope_and_position(
        data, rotary, pt, pos_ids_pt, dims, x_pt.dtype,
    )
    attn_mask = None
    if use_causal_mask:
        attn_mask = _causal_additive_mask_torch(
            dims.batch, dims.seq, x_pt.device,
        )
        data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
    out = _gptneox_attn_forward(pt, x_pt, cos, sin, attn_mask=attn_mask)
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
    pt = PtLayer(config, layer_idx=0)
    pt.eval()
    rotary = GPTNeoXRotaryEmbedding(config, device=torch.device("cpu"))
    data = _gptneox_decoder_weights(pt, "decoder", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    pos_ids_pt = _position_ids_pt(dims, x_pt.device)
    cos, sin = _write_rope_and_position(
        data, rotary, pt.attention, pos_ids_pt, dims, x_pt.dtype,
    )
    attn_mask = _causal_additive_mask_torch(
        dims.batch, dims.seq, x_pt.device,
    )
    data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
    out = _gptneox_decoder_forward(
        pt, x_pt, cos, sin, attn_mask=attn_mask,
    )
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data


def generate_model(seed: int, dims: TestDims = MODEL_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtModel(config)
    pt.eval()
    rotary = pt.rotary_emb
    data = _model_weights(pt, "model", dims)
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt
    data["position_ids"] = _position_ids(dims)
    data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
    pos_ids_pt = _position_ids_pt(dims, ids_pt.device)
    embeds = pt.embed_in(ids_pt)
    cos, sin = rotary(embeds, pos_ids_pt)
    cos_np, sin_np = _rope_half_from_hf(cos, sin, dims)
    data["rope_cos"] = fortran_order(cos_np)
    data["rope_sin"] = fortran_order(sin_np)
    attn_mask = _causal_additive_mask_torch(
        dims.batch, dims.seq, ids_pt.device,
    )
    out = _gptneox_model_forward(
        pt, ids_pt, cos, sin, attn_mask=attn_mask,
    )
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_embed_tokens_vocab"] = fortran_order(
        pt.embed_in.weight.grad.detach().numpy().T)
    return data


def generate_causal(seed: int, dims: TestDims = CAUSAL_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtCausalLM(config)
    pt.eval()
    rotary = pt.gpt_neox.rotary_emb
    data = _model_weights(pt.gpt_neox, "model.model", dims)
    data["model.lm_head.weight"] = _lm_head_to_linear_weight(pt.embed_out)
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt
    data["position_ids"] = _position_ids(dims)
    data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
    pos_ids_pt = _position_ids_pt(dims, ids_pt.device)
    embeds = pt.gpt_neox.embed_in(ids_pt)
    cos, sin = rotary(embeds, pos_ids_pt)
    cos_np, sin_np = _rope_half_from_hf(cos, sin, dims)
    data["rope_cos"] = fortran_order(cos_np)
    data["rope_sin"] = fortran_order(sin_np)
    attn_mask = _causal_additive_mask_torch(
        dims.batch, dims.seq, ids_pt.device,
    )
    hidden = _gptneox_model_forward(
        pt.gpt_neox, ids_pt, cos, sin, attn_mask=attn_mask,
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
        write_fixture_json(out, stem, MLP_DIMS, 2e-5, 2e-5)
    elif args.block in ("attention", "attention_causal"):
        # SDPA/RoPE vs PyTorch eager can differ slightly at tight tol.
        write_fixture_json(out, stem, ATTENTION_DIMS, 5e-3, 5e-3)
    elif args.block == "decoder":
        # Full decoder (LN + attn + MLP + parallel residual) vs eager PyTorch.
        write_fixture_json(out, stem, DECODER_DIMS, 2e-1, 2e-1)
    elif args.block in ("model", "causal"):
        write_fixture_json(out, stem, MODEL_DIMS, 2e-1, 2e-1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
