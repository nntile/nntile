#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file nntile/tests/model/roberta/generate_test_data.py
# Generate RoBERTa building-block test data in safetensors format.
#
# @version 1.2.0

"""Generate reference test data for NNTile RoBERTa graph C++ tests.

For each block the script creates ``roberta_<block>.safetensors`` plus a paired
``.json`` sidecar (geometry, tolerances) read by the corresponding C++ tests.

Uses HuggingFace ``modeling_roberta`` for all forward/backward references
(``RobertaIntermediate``, ``RobertaAttention``, ``RobertaLayer``,
``RobertaEmbeddings``, ``RobertaModel``, ``RobertaForMaskedLM``,
``RobertaLMHead``) plus NumPy layout helpers shared with the BERT generator.
Weight tensors are reshaped to the graph module layout; reference forwards call
HF modules only (no custom RoBERTa reimplementation). Position ids follow
``create_position_ids_from_input_ids`` from the same HF file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import RobertaConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention as PtAttention, RobertaEmbeddings as PtEmbeddings,
    RobertaForMaskedLM as PtMlm, RobertaIntermediate as PtIntermediate,
    RobertaLayer as PtLayer, RobertaModel as PtModel,
    create_position_ids_from_input_ids)


def _load_bert_generate_test_data():
    """Load BERT safetensor layout helpers for shared encoder blocks."""
    path = (
        Path(__file__).resolve().parent.parent
        / "bert"
        / "generate_test_data.py"
    )
    name = "nntile_bert_generate_test_data"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"cannot import BERT test data module: {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bert_data = _load_bert_generate_test_data()

fortran_order = bert_data.fortran_order
fortran_order_int64 = bert_data.fortran_order_int64
_linear = bert_data._linear
_layer_norm = bert_data._layer_norm
_embed = bert_data._embed
_encoder_attention_weights = bert_data._bert_attention_weights
_encoder_layer_weights = bert_data._bert_layer_weights
_hidden_input = bert_data._hidden_input
_grad_output = bert_data._grad_output
_out_to_nntile = bert_data._out_to_nntile


@dataclass
class TestDims:
    hidden: int
    intermediate: int
    n_heads: int
    seq: int
    batch: int
    vocab: int
    num_layers: int
    pad_token_id: int = 1
    layer_norm_eps: float = 1e-5

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


def _make_config(dims: TestDims) -> RobertaConfig:
    return RobertaConfig(
        hidden_size=dims.hidden,
        intermediate_size=dims.intermediate,
        num_attention_heads=dims.n_heads,
        num_hidden_layers=dims.num_layers,
        vocab_size=dims.vocab,
        max_position_embeddings=max(dims.seq * 2, 128),
        pad_token_id=dims.pad_token_id,
        layer_norm_eps=dims.layer_norm_eps,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        _attn_implementation="eager",
    )


def _zero_token_type_embeddings(pt_embeddings: PtEmbeddings) -> None:
    # NNTile RobertaEmbeddings omits token-type; HF still adds
    # token_type_embeddings when token_type_ids default to zero.
    pt_embeddings.token_type_embeddings.weight.data.zero_()


def _ids_input(rng, dims: TestDims):
    low = dims.pad_token_id + 1
    ids = rng.integers(
        low,
        dims.vocab,
        size=(dims.seq, dims.batch),
    ).astype(np.int64)
    ids_nt = ids.ravel("F").reshape(ids.shape)
    ids_pt = torch.tensor(ids.T.copy(), dtype=torch.long)
    return ids_nt, ids_pt


def _position_ids_from_input_ids(
    ids_pt: torch.Tensor,
    padding_idx: int,
) -> tuple[np.ndarray, torch.Tensor]:
    """HF RoBERTa position ids (``create_position_ids_from_input_ids``)."""
    pos_pt = create_position_ids_from_input_ids(ids_pt, padding_idx)
    pos_nt = fortran_order_int64(pos_pt.detach().cpu().numpy().T)
    return pos_nt, pos_pt


def _roberta_fixture_json(
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
        "roberta": {
            "vocab_size": dims.vocab,
            "hidden_size": dims.hidden,
            "intermediate_size": dims.intermediate,
            "num_hidden_layers": dims.num_layers,
            "num_attention_heads": dims.n_heads,
            "max_position_embeddings": max(dims.seq * 2, 128),
            "pad_token_id": dims.pad_token_id,
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
            _roberta_fixture_json(stem, dims, forward_tol, backward_tol),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {path}")


def generate_intermediate(
    seed: int,
    dims: TestDims = INTERMEDIATE_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtIntermediate(config)
    pt.eval()
    data = _linear(pt.dense, "intermediate.dense")
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    out = pt(x_pt)
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data


def generate_attention(
    seed: int,
    dims: TestDims = ATTENTION_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtAttention(config)
    pt.eval()
    data = _encoder_attention_weights(pt, "attn", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    out = pt(x_pt)[0]
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data


def generate_layer(
    seed: int,
    dims: TestDims = LAYER_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtLayer(config)
    pt.eval()
    data = _encoder_layer_weights(pt, "layer", dims)
    x_nt, x_pt = _hidden_input(rng, dims)
    data["input"] = x_nt
    out = pt(x_pt)[0]
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_input"] = _out_to_nntile(x_pt.grad)
    return data


def generate_embeddings(
    seed: int,
    dims: TestDims = EMBEDDINGS_DIMS,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtEmbeddings(config)
    pt.eval()
    _zero_token_type_embeddings(pt)
    data = {}
    data.update(_embed(pt.word_embeddings, "embeddings.word"))
    data.update(_embed(pt.position_embeddings, "embeddings.position"))
    data.update(_layer_norm(pt.LayerNorm, "embeddings.ln"))
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt
    pos_nt, pos_pt = _position_ids_from_input_ids(ids_pt, dims.pad_token_id)
    data["position_ids"] = pos_nt
    out = pt(input_ids=ids_pt, position_ids=pos_pt)
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_wte_vocab"] = fortran_order(
        pt.word_embeddings.weight.grad.detach().numpy().T
    )
    return data


def _model_weights(model: PtModel, prefix: str, dims: TestDims):
    d = {}
    d.update(
        _embed(model.embeddings.word_embeddings, f"{prefix}.embeddings.word")
    )
    d.update(
        _embed(
            model.embeddings.position_embeddings,
            f"{prefix}.embeddings.position",
        )
    )
    d.update(
        _layer_norm(model.embeddings.LayerNorm, f"{prefix}.embeddings.ln")
    )
    for i, layer in enumerate(model.encoder.layer):
        d.update(_encoder_layer_weights(layer, f"{prefix}.layer_{i}", dims))
    return d


def generate_model(seed: int, dims: TestDims = MODEL_DIMS):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtModel(config, add_pooling_layer=False)
    pt.eval()
    _zero_token_type_embeddings(pt.embeddings)
    data = _model_weights(pt, "model", dims)
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt
    pos_nt, pos_pt = _position_ids_from_input_ids(ids_pt, dims.pad_token_id)
    data["position_ids"] = pos_nt
    out = pt(input_ids=ids_pt, position_ids=pos_pt)
    data["output_ref"] = _out_to_nntile(out.last_hidden_state)
    g_nt, g_pt = _grad_output(rng, out.last_hidden_state)
    data["grad_output"] = g_nt
    out.last_hidden_state.backward(g_pt)
    data["grad_wte_vocab"] = fortran_order(
        pt.embeddings.word_embeddings.weight.grad.detach().numpy().T
    )
    return data


def _roberta_mlm_head_weights(head, prefix: str):
    d = {}
    d.update(_linear(head.dense, f"{prefix}.transform_dense"))
    d.update(_layer_norm(head.layer_norm, f"{prefix}.transform_ln"))
    d.update(_linear(head.decoder, f"{prefix}.decoder"))
    d[f"{prefix}.head_bias"] = fortran_order(head.bias.detach().numpy())
    return d


def generate_mlm(seed: int, dims: TestDims = MLM_DIMS):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(dims)
    pt = PtMlm(config)
    pt.eval()
    _zero_token_type_embeddings(pt.roberta.embeddings)
    data = _model_weights(pt.roberta, "model.roberta", dims)
    data.update(_roberta_mlm_head_weights(pt.lm_head, "model.cls"))
    ids_nt, ids_pt = _ids_input(rng, dims)
    data["input_ids"] = ids_nt
    pos_nt, pos_pt = _position_ids_from_input_ids(ids_pt, dims.pad_token_id)
    data["position_ids"] = pos_nt
    out = pt(input_ids=ids_pt, position_ids=pos_pt).logits
    data["output_ref"] = _out_to_nntile(out)
    g_nt, g_pt = _grad_output(rng, out)
    data["grad_output"] = g_nt
    out.backward(g_pt)
    data["grad_wte_vocab"] = fortran_order(
        pt.roberta.embeddings.word_embeddings.weight.grad.detach().numpy().T
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate RoBERTa block test data (safetensors)",
    )
    parser.add_argument(
        "--block",
        choices=GENERATORS,
        required=True,
        help="RoBERTa block to generate data for",
    )
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--seed", "-s", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    data = GENERATORS[args.block](args.seed)
    stem = f"roberta_{args.block}"
    bundle_path = str(out / f"{stem}.safetensors")
    save_file(data, bundle_path)
    print(f"Saved {bundle_path}")

    tol = 1e-6
    if args.block == "intermediate":
        write_fixture_json(out, stem, INTERMEDIATE_DIMS, tol, tol)
    elif args.block == "attention":
        write_fixture_json(out, stem, ATTENTION_DIMS, tol, tol)
    elif args.block == "layer":
        write_fixture_json(out, stem, LAYER_DIMS, tol, tol)
    elif args.block == "embeddings":
        write_fixture_json(out, stem, EMBEDDINGS_DIMS, tol, tol)
    elif args.block in ("model", "mlm"):
        write_fixture_json(out, stem, MODEL_DIMS, tol, tol)

    return 0


if __name__ == "__main__":
    sys.exit(main())
