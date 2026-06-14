#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file nntile/tests/model/bert/generate_test_data.py
# Generate BERT building-block test data in safetensors format.
#
# @version 1.1.0

"""Generate reference test data for NNTile BERT graph C++ tests.

For each block the script creates ``bert_<block>.safetensors`` plus a paired
``.json`` sidecar (geometry, tolerances) read by the corresponding C++ tests.

All forward and backward references come from HuggingFace ``modeling_bert``
(PyTorch eager, dropout disabled). Helpers below only reshape HF parameters
into NNTile C-order layouts expected by the graph API modules; they do
not reimplement BERT computation.
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
from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertAttention as PtAttention, BertEmbeddings as PtEmbeddings,
    BertForMaskedLM as PtMlm, BertIntermediate as PtIntermediate,
    BertLayer as PtLayer, BertModel as PtModel)

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
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12

    @property
    def head_size(self) -> int:
        return self.hidden // self.n_heads


INTERMEDIATE_DIMS = TestDims(
    hidden=8,
    intermediate=16,
    n_heads=4,
    seq=4,
    batch=2,
    vocab=100,
    num_layers=1,
)

ATTENTION_DIMS = TestDims(
    hidden=64,
    intermediate=256,
    n_heads=4,
    seq=8,
    batch=2,
    vocab=100,
    num_layers=1,
)

LAYER_DIMS = ATTENTION_DIMS
EMBEDDINGS_DIMS = ATTENTION_DIMS
MODEL_DIMS = ATTENTION_DIMS
MLM_DIMS = ATTENTION_DIMS

# ── Layout helpers (NumPy / PyTorch ↔ NNTile) ─────────────────────────────


def as_float32(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float32)


def as_int64(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.int64)


def _make_config(dims: TestDims) -> BertConfig:
    return BertConfig(
        hidden_size=dims.hidden,
        intermediate_size=dims.intermediate,
        num_attention_heads=dims.n_heads,
        num_hidden_layers=dims.num_layers,
        vocab_size=dims.vocab,
        max_position_embeddings=max(dims.seq * 2, 128),
        type_vocab_size=dims.type_vocab_size,
        layer_norm_eps=dims.layer_norm_eps,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        _attn_implementation="eager",
    )


def _layer_norm(ln, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.gamma": as_float32(ln.weight.detach().numpy()),
        f"{prefix}.beta": as_float32(ln.bias.detach().numpy()),
    }


def _linear(linear, prefix: str) -> dict[str, np.ndarray]:
    # PyTorch Linear weight is (out_features, in_features); graph Linear
    # stores the same C-order layout (output_dim, input_dim).
    d = {
        f"{prefix}.weight": as_float32(linear.weight.detach().numpy()),
    }
    if linear.bias is not None:
        d[f"{prefix}.bias"] = as_float32(linear.bias.detach().numpy())
    return d


def _embed(embed, prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}.vocab": as_float32(embed.weight.detach().numpy())}


def _bert_self_attn_weights(self_attn, prefix: str, dims: TestDims):
    """Split HF BertSelfAttention Linear weights into graph head layouts."""
    n_emb = dims.hidden
    hs = dims.head_size
    n_heads = dims.n_heads

    def w(linear):
        return linear.weight.detach().numpy().reshape(
            n_heads, hs, n_emb,
        ).transpose(2, 1, 0)

    def b(linear):
        return linear.bias.detach().numpy().reshape(n_heads, hs).transpose(1, 0)

    return {
        f"{prefix}.q_weight": as_float32(w(self_attn.query)),
        f"{prefix}.k_weight": as_float32(w(self_attn.key)),
        f"{prefix}.v_weight": as_float32(w(self_attn.value)),
        f"{prefix}.q_bias": as_float32(b(self_attn.query)),
        f"{prefix}.k_bias": as_float32(b(self_attn.key)),
        f"{prefix}.v_bias": as_float32(b(self_attn.value)),
    }


def _bert_self_output_weights(out_module, prefix: str, dims: TestDims):
    n_emb = dims.hidden
    n_heads = dims.n_heads
    hs = dims.head_size
    w = (
        out_module.dense.weight.detach()
        .numpy()
        .reshape(n_heads, hs, n_emb)
        .transpose(1, 0, 2)
    )
    return {
        f"{prefix}.dense.weight": as_float32(w),
        f"{prefix}.dense.bias": as_float32(
            out_module.dense.bias.detach().numpy(),
        ),
        **_layer_norm(out_module.LayerNorm, f"{prefix}.ln"),
    }


def _bert_attention_weights(attn: PtAttention, prefix: str, dims: TestDims):
    d: dict[str, np.ndarray] = {}
    d.update(_bert_self_attn_weights(attn.self, f"{prefix}.self", dims))
    d.update(_bert_self_output_weights(attn.output, f"{prefix}.output", dims))
    return d


def _bert_layer_weights(layer: PtLayer, prefix: str, dims: TestDims):
    d: dict[str, np.ndarray] = {}
    d.update(
        _bert_attention_weights(
            layer.attention,
            f"{prefix}.attention",
            dims,
        ),
    )
    d.update(_linear(layer.intermediate.dense, f"{prefix}.intermediate.dense"))
    d.update(_linear(layer.output.dense, f"{prefix}.output.dense"))
    d.update(_layer_norm(layer.output.LayerNorm, f"{prefix}.output.ln"))
    return d


def _hidden_input(rng, dims: TestDims, scale: float = 0.1):
    x = rng.standard_normal(
        (dims.batch, dims.seq, dims.hidden),
    ).astype(np.float32) * scale
    x_nt = as_float32(x)
    x_pt = torch.tensor(x.copy(), requires_grad=True)
    return x_nt, x_pt


def _grad_output(rng, pt_out: torch.Tensor, scale: float = 0.1):
    g = rng.standard_normal(pt_out.shape).astype(np.float32) * scale
    g_pt = torch.tensor(g)
    g_nt = as_float32(g)
    return g_nt, g_pt


def _ids_input(rng, dims: TestDims):
    ids = rng.integers(
        0,
        dims.vocab,
        size=(dims.batch, dims.seq),
    ).astype(np.int64)
    ids_nt = as_int64(ids)
    ids_pt = torch.tensor(ids.copy(), dtype=torch.long)
    return ids_nt, ids_pt


def _token_type_ids(dims: TestDims) -> np.ndarray:
    tt = np.zeros((dims.batch, dims.seq), dtype=np.int64)
    return as_int64(tt)


def _position_ids(dims: TestDims) -> np.ndarray:
    pos = np.arange(dims.seq, dtype=np.int64)[None, :]
    pos = np.broadcast_to(pos, (dims.batch, dims.seq)).copy()
    return as_int64(pos)


def _bert_batch_inputs(dims: TestDims):
    """HF BertEmbeddings/BertModel inputs: (batch, seq) ids and masks."""
    tt_pt = torch.zeros(dims.batch, dims.seq, dtype=torch.long)
    pos_pt = (
        torch.arange(dims.seq, dtype=torch.long)
        .unsqueeze(0)
        .expand(
            dims.batch,
            -1,
        )
    )
    return tt_pt, pos_pt


def _out_to_nntile(pt_out: torch.Tensor) -> np.ndarray:
    return as_float32(pt_out.detach().numpy())


def _run_hidden_block(
    make_pt,
    weight_fn,
    prefix: str,
    dims: TestDims,
    seed: int,
) -> dict[str, np.ndarray]:
    """Forward/backward through a single HF block on random hidden states."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pt_module = make_pt()
    pt_module.eval()
    data = weight_fn(pt_module, prefix, dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    out = pt_module(x_pt)
    if isinstance(out, tuple):
        out = out[0]
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data


# ── Fixture metadata ──────────────────────────────────────────────────────


def _bert_fixture_json(
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
        "bert": {
            "vocab_size": dims.vocab,
            "hidden_size": dims.hidden,
            "intermediate_size": dims.intermediate,
            "num_hidden_layers": dims.num_layers,
            "num_attention_heads": dims.n_heads,
            "max_position_embeddings": max(dims.seq * 2, 128),
            "type_vocab_size": dims.type_vocab_size,
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
            _bert_fixture_json(stem, dims, forward_tol, backward_tol),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {path}")


# ── Block generators (HF forward / backward) ──────────────────────────────


def generate_intermediate(
    seed: int,
    dims: TestDims = INTERMEDIATE_DIMS,
) -> dict[str, np.ndarray]:
    config = _make_config(dims)
    return _run_hidden_block(
        lambda: PtIntermediate(config),
        lambda m, _p, _d: _linear(m.dense, "intermediate.dense"),
        "intermediate",
        dims,
        seed,
    )


def generate_attention(
    seed: int,
    dims: TestDims = ATTENTION_DIMS,
) -> dict[str, np.ndarray]:
    config = _make_config(dims)
    return _run_hidden_block(
        lambda: PtAttention(config),
        _bert_attention_weights,
        "attn",
        dims,
        seed,
    )


def generate_layer(
    seed: int,
    dims: TestDims = LAYER_DIMS,
) -> dict[str, np.ndarray]:
    config = _make_config(dims)
    return _run_hidden_block(
        lambda: PtLayer(config),
        _bert_layer_weights,
        "layer",
        dims,
        seed,
    )


def generate_embeddings(
    seed: int,
    dims: TestDims = EMBEDDINGS_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtEmbeddings(config)
    pt.eval()
    data = {}
    data.update(_embed(pt.word_embeddings, "embeddings.word"))
    data.update(_embed(pt.position_embeddings, "embeddings.position"))
    data.update(_embed(pt.token_type_embeddings, "embeddings.token_type"))
    data.update(_layer_norm(pt.LayerNorm, "embeddings.ln"))
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt
    data["token_type_ids"] = _token_type_ids(dims)
    data["position_ids"] = _position_ids(dims)
    tt_pt, pos_pt = _bert_batch_inputs(dims)
    out = pt(ids_pt, tt_pt, pos_pt)
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_wte_vocab"] = as_float32(
        pt.word_embeddings.weight.grad.detach().numpy(),
    )
    return data


def _model_weights(model: PtModel, prefix: str, dims: TestDims):
    d: dict[str, np.ndarray] = {}
    d.update(
        _embed(
            model.embeddings.word_embeddings,
            f"{prefix}.embeddings.word",
        ),
    )
    d.update(
        _embed(
            model.embeddings.position_embeddings,
            f"{prefix}.embeddings.position",
        ),
    )
    d.update(
        _embed(
            model.embeddings.token_type_embeddings,
            f"{prefix}.embeddings.token_type",
        ),
    )
    d.update(
        _layer_norm(
            model.embeddings.LayerNorm,
            f"{prefix}.embeddings.ln",
        ),
    )
    for i, layer in enumerate(model.encoder.layer):
        d.update(_bert_layer_weights(layer, f"{prefix}.layer_{i}", dims))
    return d


def _mlm_head_weights(
    head,
    prefix: str,
    word_embeddings,
) -> dict[str, np.ndarray]:
    """HF ties MLM decoder to word embeddings; graph BertMlmHead mirrors."""
    if head.decoder.weight is not word_embeddings.weight:
        raise RuntimeError(
            "BertLMPredictionHead decoder weight must be tied to "
            "word embeddings",
        )
    d: dict[str, np.ndarray] = {}
    d.update(_linear(head.transform.dense, f"{prefix}.transform_dense"))
    d.update(_layer_norm(head.transform.LayerNorm, f"{prefix}.transform_ln"))
    d.update(_linear(head.decoder, f"{prefix}.decoder"))
    return d


def generate_model(seed: int, dims: TestDims = MODEL_DIMS):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtModel(config, add_pooling_layer=False)
    pt.eval()
    data = _model_weights(pt, "model", dims)
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt
    data["token_type_ids"] = _token_type_ids(dims)
    data["position_ids"] = _position_ids(dims)
    tt_pt, pos_pt = _bert_batch_inputs(dims)
    out = pt(input_ids=ids_pt, token_type_ids=tt_pt, position_ids=pos_pt)
    data["output_ref"] = _out_to_nntile(out.last_hidden_state)
    g_nt, g_pt = _grad_output(rng, out.last_hidden_state)
    data["grad_output"] = g_nt
    out.last_hidden_state.backward(g_pt)
    data["grad_wte_vocab"] = as_float32(
        pt.embeddings.word_embeddings.weight.grad.detach().numpy(),
    )
    return data


def generate_mlm(seed: int, dims: TestDims = MLM_DIMS):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtMlm(config)
    pt.eval()
    word_emb = pt.bert.embeddings.word_embeddings
    data = _model_weights(pt.bert, "model.bert", dims)
    data.update(
        _mlm_head_weights(pt.cls.predictions, "model.cls", word_emb),
    )
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt
    data["token_type_ids"] = _token_type_ids(dims)
    data["position_ids"] = _position_ids(dims)
    tt_pt, pos_pt = _bert_batch_inputs(dims)
    out = pt(
        input_ids=ids_pt,
        token_type_ids=tt_pt,
        position_ids=pos_pt,
    ).logits
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_wte_vocab"] = as_float32(
        word_emb.weight.grad.detach().numpy(),
    )
    return data


GENERATORS = {
    "intermediate": generate_intermediate,
    "attention": generate_attention,
    "layer": generate_layer,
    "embeddings": generate_embeddings,
    "model": generate_model,
    "mlm": generate_mlm,
}

BLOCK_DIMS = {
    "intermediate": INTERMEDIATE_DIMS,
    "attention": ATTENTION_DIMS,
    "layer": LAYER_DIMS,
    "embeddings": EMBEDDINGS_DIMS,
    "model": MODEL_DIMS,
    "mlm": MLM_DIMS,
}

DEFAULT_TOL = 1e-6


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate BERT block test data (safetensors)",
    )
    parser.add_argument(
        "--block",
        choices=GENERATORS,
        required=True,
        help="BERT block to generate data for",
    )
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--seed", "-s", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    data = GENERATORS[args.block](args.seed)
    stem = f"bert_{args.block}"
    bundle_path = str(out / f"{stem}.safetensors")
    save_file(data, bundle_path)
    print(f"Saved {bundle_path}")

    dims = BLOCK_DIMS[args.block]
    write_fixture_json(out, stem, dims, DEFAULT_TOL, DEFAULT_TOL)

    return 0


if __name__ == "__main__":
    sys.exit(main())
