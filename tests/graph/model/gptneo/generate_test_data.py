#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file tests/graph/model/gptneo/generate_test_data.py
# Generate GPT-Neo building-block test data in safetensors format.
#
# @version 1.1.0

"""Generate reference test data for NNTile GPT-Neo graph C++ tests.

For each block the script creates ``gptneo_<block>.safetensors`` plus a paired
``.json`` sidecar (geometry, tolerances) read by the corresponding C++ tests.

Uses HuggingFace ``modeling_gpt_neo`` with NNTile layout per
``examples/gptneo_generate.py``. Reference forwards use HF LayerNorm (gamma/beta), GELUTANH for MLP, and
separate Q/K/V/O projections with ``out_proj`` bias (add_fiber).
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
from transformers import GPTNeoConfig
from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoAttention as PtAttention,
    GPTNeoBlock as PtBlock,
    GPTNeoForCausalLM as PtCausalLM,
    GPTNeoMLP as PtMLP,
    GPTNeoModel as PtModel,
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


def _make_config(dims: TestDims) -> GPTNeoConfig:
    return GPTNeoConfig(
        vocab_size=dims.vocab,
        hidden_size=dims.hidden,
        num_layers=dims.num_layers,
        num_heads=dims.n_heads,
        intermediate_size=dims.intermediate,
        max_position_embeddings=max(dims.seq * 2, 128),
        layer_norm_epsilon=dims.layer_norm_eps,
        attention_types=[[["global"], dims.num_layers]],
        resid_dropout=0.0,
        embed_dropout=0.0,
        attention_dropout=0.0,
        _attn_implementation="eager",
    )


def _layer_norm(ln, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.gamma": fortran_order(ln.weight.detach().numpy()),
        f"{prefix}.beta": fortran_order(ln.bias.detach().numpy()),
    }


def _gelutanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def _gptneo_attn_weights(
    attn: PtAttention, prefix: str, dims: TestDims,
) -> dict[str, np.ndarray]:
    """Map HF q/k/v/out_proj to NNTile layouts (``examples/gptneo_generate.py``)."""
    inner = attn.attention
    H = dims.hidden
    nh = dims.n_heads
    hd = dims.head_size
    return {
        f"{prefix}.q_weight": fortran_order(
            inner.q_proj.weight.detach().numpy().reshape(nh, hd, H)),
        f"{prefix}.k_weight": fortran_order(
            inner.k_proj.weight.detach().numpy().reshape(nh, hd, H)),
        f"{prefix}.v_weight": fortran_order(
            inner.v_proj.weight.detach().numpy().reshape(nh, hd, H)),
        f"{prefix}.o_weight": fortran_order(
            inner.out_proj.weight.detach().numpy().reshape(H, nh, hd)),
        f"{prefix}.o_bias": fortran_order(
            inner.out_proj.bias.detach().numpy()),
    }


def _gptneo_mlp_weights(mlp: PtMLP, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.fc1.weight": fortran_order(
            mlp.c_fc.weight.detach().numpy().T),
        f"{prefix}.fc2.weight": fortran_order(
            mlp.c_proj.weight.detach().numpy().T),
    }


def _gptneo_decoder_weights(
    layer: PtBlock, prefix: str, dims: TestDims,
) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {}
    d.update(_layer_norm(layer.ln_1, f"{prefix}.input_norm"))
    d.update(_gptneo_attn_weights(layer.attn, f"{prefix}.self_attn", dims))
    d.update(_layer_norm(layer.ln_2, f"{prefix}.post_attn_norm"))
    d.update(_gptneo_mlp_weights(layer.mlp, f"{prefix}.mlp"))
    return d


def _embed(embed, prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}.vocab": fortran_order(embed.weight.detach().numpy().T)}


def _model_weights(model: PtModel, prefix: str, dims: TestDims) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {}
    d.update(_embed(model.wte, f"{prefix}.wte"))
    d.update(_embed(model.wpe, f"{prefix}.wpe"))
    d.update(_layer_norm(model.ln_f, f"{prefix}.norm"))
    for i, layer in enumerate(model.h):
        d.update(_gptneo_decoder_weights(layer, f"{prefix}.layers_{i}", dims))
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


def _out_to_nntile(pt_out: torch.Tensor) -> np.ndarray:
    return fortran_order(pt_out.detach().numpy().transpose(2, 1, 0))


def _sdpa_causal_mask_fortran(seq: int) -> np.ndarray:
    kk = np.arange(seq, dtype=np.int64)[:, None]
    qq = np.arange(seq, dtype=np.int64)[None, :]
    allowed = (kk <= qq).astype(np.float32)
    return fortran_order(allowed)


def _sdpa_gptneo_local_mask_fortran(seq: int, window: int) -> np.ndarray:
    kk = np.arange(seq, dtype=np.int64)[:, None]
    qq = np.arange(seq, dtype=np.int64)[None, :]
    allowed = ((kk <= qq) & (qq - kk < window)).astype(np.float32)
    return fortran_order(allowed)


def _local_additive_mask_torch(
    batch: int, seq: int, window: int, device: torch.device,
) -> torch.Tensor:
    allowed = _sdpa_gptneo_local_mask_fortran(seq, window)
    block = (1.0 - allowed) * torch.finfo(torch.float32).min
    mask_torch = torch.tensor(block, device=device, dtype=torch.float32)
    return mask_torch[None, None, :, :].expand(batch, 1, -1, -1)


def _causal_additive_mask_torch(
    batch: int, seq: int, device: torch.device,
) -> torch.Tensor:
    mask = np.array(np.triu(np.ones((seq, seq))), dtype=bool, order="F")
    mask_torch = torch.tensor(
        np.array(1 - mask, dtype=np.float32),
    ).T * torch.finfo(torch.float32).min
    mask_torch = mask_torch.to(device=device, dtype=torch.float32)
    return mask_torch[None, None, :, :].expand(batch, 1, -1, -1)


def _gptneo_mlp_forward(mlp: PtMLP, x_pt: torch.Tensor) -> torch.Tensor:
    """Bias-free MLP forward (matches graph ``GptneoMLP``)."""
    h = F.linear(x_pt, mlp.c_fc.weight, None)
    h = _gelutanh(h)
    return F.linear(h, mlp.c_proj.weight, None)


def _gptneo_attn_forward(
    attn: PtAttention,
    x_pt: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Q/K/V + SDPA + out_proj + bias (matches graph GptneoAttention)."""
    inner = attn.attention
    n_emb = x_pt.shape[-1]
    n_head = inner.num_heads
    head_dim = inner.head_dim
    q = inner.q_proj(x_pt)
    k = inner.k_proj(x_pt)
    v = inner.v_proj(x_pt)
    shape = (*x_pt.shape[:2], n_head, head_dim)
    q = q.view(*shape).transpose(1, 2)
    k = k.view(*shape).transpose(1, 2)
    v = v.view(*shape).transpose(1, 2)
    ctx = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        is_causal=False,
    )
    ctx = ctx.transpose(1, 2).contiguous().view(*x_pt.shape)
    return inner.out_proj(ctx)


def _gptneo_decoder_forward(
    block: PtBlock,
    x_pt: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """HF LayerNorm + graph-aligned attention/MLP (matches C++ GptneoDecoder)."""
    residual = x_pt
    x_norm = block.ln_1(x_pt)
    attn_out = _gptneo_attn_forward(block.attn, x_norm, attn_mask=attn_mask)
    post_attn = residual + attn_out
    mlp_in = block.ln_2(post_attn)
    mlp_out = _gptneo_mlp_forward(block.mlp, mlp_in)
    return post_attn + mlp_out


def _gptneo_model_forward(
    model: PtModel,
    ids_pt: torch.Tensor,
    pos_pt: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    x = model.wte(ids_pt) + model.wpe(pos_pt)
    for layer in model.h:
        x = _gptneo_decoder_forward(layer, x, attn_mask=attn_mask)
    return model.ln_f(x)


def _gptneo_fixture_json(
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
        "gptneo": {
            "vocab_size": dims.vocab,
            "hidden_size": dims.hidden,
            "intermediate_size": dims.intermediate,
            "num_hidden_layers": dims.num_layers,
            "num_attention_heads": dims.n_heads,
            "head_dim": dims.head_size,
            "max_position_embeddings": max(dims.seq * 2, 128),
            "layer_norm_eps": dims.layer_norm_eps,
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
            _gptneo_fixture_json(stem, dims, forward_tol, backward_tol),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {path}")


def generate_mlp(seed: int, dims: TestDims = MLP_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtMLP(dims.intermediate, config)
    pt.eval()
    data = _gptneo_mlp_weights(pt, "mlp")
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    out = _gptneo_mlp_forward(pt, x_pt)
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
    pt = PtAttention(config, layer_id=0)
    pt.eval()
    data = _gptneo_attn_weights(pt, "attn", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    if use_causal_mask:
        attn_mask = _causal_additive_mask_torch(
            dims.batch, dims.seq, x_pt.device,
        )
        data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
        out = _gptneo_attn_forward(pt, x_pt, attn_mask=attn_mask)
    else:
        out = _gptneo_attn_forward(pt, x_pt, attn_mask=None)
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data




def generate_attention_local(
    seed: int, dims: TestDims = ATTENTION_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = GPTNeoConfig(
        vocab_size=dims.vocab,
        hidden_size=dims.hidden,
        num_layers=2,
        num_heads=dims.n_heads,
        intermediate_size=dims.intermediate,
        max_position_embeddings=max(dims.seq * 2, 128),
        layer_norm_epsilon=dims.layer_norm_eps,
        attention_types=[[["global"], 1], [["local"], 1]],
        resid_dropout=0.0,
        embed_dropout=0.0,
        attention_dropout=0.0,
        _attn_implementation="eager",
    )
    pt = PtAttention(config, layer_id=1)
    pt.eval()
    data = _gptneo_attn_weights(pt, "attn", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    window = int(config.window_size)
    data["attn_mask"] = _sdpa_gptneo_local_mask_fortran(dims.seq, window)
    attn_mask = _local_additive_mask_torch(
        dims.batch, dims.seq, window, x_pt.device,
    )
    out = _gptneo_attn_forward(pt, x_pt, attn_mask=attn_mask)
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
    pt = PtBlock(config, layer_id=0)
    pt.eval()
    data = _gptneo_decoder_weights(pt, "decoder", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    attn_mask = _causal_additive_mask_torch(
        dims.batch, dims.seq, x_pt.device,
    )
    data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
    out = _gptneo_decoder_forward(pt, x_pt, attn_mask=attn_mask)
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
    data = _model_weights(pt, "model", dims)
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt
    data["position_ids"] = _position_ids(dims)
    data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
    pos_pt = torch.arange(
        dims.seq, dtype=torch.long,
    ).unsqueeze(0).expand(dims.batch, dims.seq)
    attn_mask = _causal_additive_mask_torch(
        dims.batch, dims.seq, ids_pt.device,
    )
    out = _gptneo_model_forward(
        pt, ids_pt, pos_pt, attn_mask=attn_mask,
    )
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    out.backward(g_pt)
    data["grad_output"] = g_nt
    data["grad_wte_vocab"] = fortran_order(
        pt.wte.weight.grad.detach().numpy().T)
    data["grad_wpe_vocab"] = fortran_order(
        pt.wpe.weight.grad.detach().numpy().T)
    return data


def generate_causal(seed: int, dims: TestDims = CAUSAL_DIMS) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtCausalLM(config)
    pt.eval()
    data = _model_weights(pt.transformer, "model.model", dims)
    data["model.lm_head.weight"] = _lm_head_to_linear_weight(pt.lm_head)
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt
    data["position_ids"] = _position_ids(dims)
    data["attn_mask"] = _sdpa_causal_mask_fortran(dims.seq)
    pos_pt = torch.arange(
        dims.seq, dtype=torch.long,
    ).unsqueeze(0).expand(dims.batch, dims.seq)
    attn_mask = _causal_additive_mask_torch(
        dims.batch, dims.seq, ids_pt.device,
    )
    hidden = _gptneo_model_forward(
        pt.transformer, ids_pt, pos_pt, attn_mask=attn_mask,
    )
    logits = pt.lm_head(hidden)
    data["output_ref"] = _out_to_nntile(logits)
    g_nt, g_pt = _grad_output(rng, logits)
    logits.backward(g_pt)
    data["grad_output"] = g_nt
    data["grad_wte_vocab"] = fortran_order(
        pt.transformer.wte.weight.grad.detach().numpy().T)
    data["grad_wpe_vocab"] = fortran_order(
        pt.transformer.wpe.weight.grad.detach().numpy().T)
    return data


GENERATORS = {
    "mlp": generate_mlp,
    "attention": lambda seed: generate_attention(seed, use_causal_mask=False),
    "attention_causal": lambda seed: generate_attention(
        seed, use_causal_mask=True,
    ),
    "attention_local": generate_attention_local,
    "decoder": generate_decoder,
    "model": generate_model,
    "causal": generate_causal,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate GPT-Neo block test data (safetensors)",
    )
    parser.add_argument(
        "--block",
        choices=GENERATORS,
        required=True,
        help="GPT-Neo block to generate data for",
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
    stem = f"gptneo_{args.block}"
    bundle_path = str(out / f"{stem}.safetensors")
    save_file(data, bundle_path)
    print(f"Saved {bundle_path}")

    if args.block == "mlp":
        write_fixture_json(out, stem, MLP_DIMS, 2e-5, 2e-5)
    elif args.block in ("attention", "attention_causal", "attention_local"):
        write_fixture_json(out, stem, ATTENTION_DIMS, 2e-5, 2e-5)
    elif args.block == "decoder":
        write_fixture_json(out, stem, DECODER_DIMS, 2e-5, 2e-5)
    elif args.block in ("model", "causal"):
        write_fixture_json(out, stem, MODEL_DIMS, 2e-5, 2e-5)

    return 0


if __name__ == "__main__":
    sys.exit(main())
