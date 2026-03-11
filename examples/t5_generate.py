#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file examples/t5_generate.py
# Convert a HuggingFace T5 checkpoint to NNTile format for C++ inference.
#
# Usage (requires T5 v1.1+ gated models, e.g. flan-t5-small):
#   python examples/t5_generate.py \
#       --model google/flan-t5-small \
#       --output-dir /tmp/nntile_t5 \
#       --encoder-prompt "translate English to German: The house is wonderful."
#
# Then run the C++ binary:
#   ./t5_generate \
#       --config /tmp/nntile_t5/config.json \
#       --weights /tmp/nntile_t5/weights.safetensors \
#       --encoder-ids "$(cat /tmp/nntile_t5/encoder_ids.txt)" \
#       --decoder-ids "$(cat /tmp/nntile_t5/decoder_ids.txt)" \
#       --max-tokens 32
#
# @version 1.1.0

"""Convert HuggingFace T5 checkpoint to NNTile format.

Produces:
  - weights.safetensors  NNTile-layout weight file
  - config.json  Model configuration readable by the C++ example
  - encoder_ids.txt  Comma-separated token IDs for encoder input
  - decoder_ids.txt  Comma-separated token IDs for decoder start
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

# ── Layout helpers ────────────────────────────────────────────────────────


def fortran_order(arr: np.ndarray) -> np.ndarray:
    """Return C-contiguous array whose flat bytes equal NNTile column-major."""
    a = np.asarray(arr, dtype=np.float32)
    return a.ravel("F").reshape(a.shape)


# ── Streaming safetensors writer ──────────────────────────────────────────


def _write_safetensors_streaming(
    path: str | Path,
    specs: Sequence[tuple[str, tuple[int, ...]]],
    get_data: Callable[[str], np.ndarray],
) -> None:
    """Write a safetensors file one tensor at a time."""
    header: dict[str, dict] = {}
    offset = 0
    entry_order: list[tuple[str, int]] = []
    for name, shape in specs:
        nbytes = int(np.prod(shape)) * 4
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        entry_order.append((name, nbytes))
        offset += nbytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for name, nbytes in entry_order:
            arr = get_data(name)
            raw = arr.tobytes()
            assert len(raw) == nbytes, (
                f"{name}: expected {nbytes} bytes, got {len(raw)}")
            f.write(raw)
            del arr, raw


# ── Output tensor specs (NNTile parameter names) ──────────────────────────


def _output_specs(config) -> list[tuple[str, tuple[int, ...]]]:
    """Return (nntile_name, shape) for every output tensor."""
    d_model = config.d_model
    d_ff = config.d_ff
    V = config.vocab_size
    n_heads = config.num_heads
    d_kv = getattr(config, "d_kv", d_model // n_heads)  # head_size

    specs: list[tuple[str, tuple[int, ...]]] = []

    # Shared embedding
    specs.append(("model.model.embed_tokens.vocab", (d_model, V)))

    # Encoder
    for i in range(config.num_layers):
        p = f"model.model.encoder_layers_{i}"
        specs.append((f"{p}.layer_norm_0.gamma", (d_model,)))
        specs.append((f"{p}.self_attn.q_weight", (n_heads, d_kv, d_model)))
        specs.append((f"{p}.self_attn.k_weight", (n_heads, d_kv, d_model)))
        specs.append((f"{p}.self_attn.v_weight", (n_heads, d_kv, d_model)))
        specs.append((f"{p}.self_attn.o_weight", (d_model, n_heads, d_kv)))
        specs.append((f"{p}.ff.layer_norm.gamma", (d_model,)))
        specs.append((f"{p}.ff.dense.gate_proj.weight", (d_model, d_ff)))
        specs.append((f"{p}.ff.dense.up_proj.weight", (d_model, d_ff)))
        specs.append((f"{p}.ff.dense.down_proj.weight", (d_ff, d_model)))

    specs.append(("model.model.encoder_final_norm.gamma", (d_model,)))

    # Decoder
    for i in range(config.num_decoder_layers):
        p = f"model.model.decoder_layers_{i}"
        specs.append((f"{p}.layer_norm_0.gamma", (d_model,)))
        specs.append((f"{p}.self_attn.q_weight", (n_heads, d_kv, d_model)))
        specs.append((f"{p}.self_attn.k_weight", (n_heads, d_kv, d_model)))
        specs.append((f"{p}.self_attn.v_weight", (n_heads, d_kv, d_model)))
        specs.append((f"{p}.self_attn.o_weight", (d_model, n_heads, d_kv)))
        specs.append((f"{p}.layer_norm_1.gamma", (d_model,)))
        specs.append((f"{p}.cross_attn.q_weight", (n_heads, d_kv, d_model)))
        specs.append((f"{p}.cross_attn.k_weight", (n_heads, d_kv, d_model)))
        specs.append((f"{p}.cross_attn.v_weight", (n_heads, d_kv, d_model)))
        specs.append((f"{p}.cross_attn.o_weight", (d_model, n_heads, d_kv)))
        specs.append((f"{p}.ff.layer_norm.gamma", (d_model,)))
        specs.append((f"{p}.ff.dense.gate_proj.weight", (d_model, d_ff)))
        specs.append((f"{p}.ff.dense.up_proj.weight", (d_model, d_ff)))
        specs.append((f"{p}.ff.dense.down_proj.weight", (d_ff, d_model)))

    specs.append(("model.model.decoder_final_norm.gamma", (d_model,)))

    # LM head
    specs.append(("model.lm_head.weight", (d_model, V)))

    return specs


# ── Per-tensor conversion ──────────────────────────────────────────────────


def _make_converter(
    config,
    hf_get: Callable[[str], np.ndarray],
    has_tensor: Callable[[str], bool],
) -> Callable[[str], np.ndarray]:
    """Return a function that converts a single NNTile tensor on demand."""
    d_model = config.d_model
    n_heads = config.num_heads
    d_kv = getattr(config, "d_kv", d_model // n_heads)

    def _get_ff_wi_2(hp: str, layer_idx: int) -> np.ndarray:
        key = f"{hp}.layer.{layer_idx}.DenseReluDense.wi_2.weight"
        if not has_tensor(key):
            raise KeyError(
                "T5 v1.0 (non-gated) models are not supported. "
                f"Missing {key}. Use T5 v1.1+ (e.g. google/flan-t5-small)."
            )
        return hf_get(key)

    def convert(name: str) -> np.ndarray:
        if name == "model.model.embed_tokens.vocab":
            return fortran_order(hf_get("shared.weight").T)

        if name == "model.model.encoder_final_norm.gamma":
            return fortran_order(hf_get("encoder.final_layer_norm.weight"))

        if name == "model.model.decoder_final_norm.gamma":
            return fortran_order(hf_get("decoder.final_layer_norm.weight"))

        if name == "model.lm_head.weight":
            return fortran_order(hf_get("lm_head.weight").T)

        # Encoder layers
        if "encoder_layers_" in name and "decoder" not in name:
            parts = name.split(".")
            layer_idx = int(parts[2].split("_", 2)[2])
            rest = ".".join(parts[3:])
            hp = f"encoder.block.{layer_idx}"

            if rest == "layer_norm_0.gamma":
                return fortran_order(
                    hf_get(f"{hp}.layer.0.layer_norm.weight"))
            if rest == "self_attn.q_weight":
                q = hf_get(f"{hp}.layer.0.SelfAttention.q.weight")
                return fortran_order(
                    q.reshape(n_heads, d_kv, d_model))
            if rest == "self_attn.k_weight":
                k = hf_get(f"{hp}.layer.0.SelfAttention.k.weight")
                return fortran_order(k.reshape(n_heads, d_kv, d_model))
            if rest == "self_attn.v_weight":
                v = hf_get(f"{hp}.layer.0.SelfAttention.v.weight")
                return fortran_order(v.reshape(n_heads, d_kv, d_model))
            if rest == "self_attn.o_weight":
                o = hf_get(f"{hp}.layer.0.SelfAttention.o.weight")
                return fortran_order(o.reshape(d_model, n_heads, d_kv))

            if rest == "ff.layer_norm.gamma":
                return fortran_order(
                    hf_get(f"{hp}.layer.1.layer_norm.weight"))
            if rest == "ff.dense.gate_proj.weight":
                return fortran_order(
                    hf_get(f"{hp}.layer.1.DenseReluDense.wi.weight").T)
            if rest == "ff.dense.up_proj.weight":
                return fortran_order(
                    _get_ff_wi_2(hp, 1).T)
            if rest == "ff.dense.down_proj.weight":
                return fortran_order(
                    hf_get(f"{hp}.layer.1.DenseReluDense.wo.weight").T)

        # Decoder layers
        if "decoder_layers_" in name:
            parts = name.split(".")
            layer_idx = int(parts[2].split("_", 2)[2])
            rest = ".".join(parts[3:])
            hp = f"decoder.block.{layer_idx}"

            if rest == "layer_norm_0.gamma":
                return fortran_order(
                    hf_get(f"{hp}.layer.0.layer_norm.weight"))
            if rest == "self_attn.q_weight":
                q = hf_get(f"{hp}.layer.0.SelfAttention.q.weight")
                return fortran_order(q.reshape(n_heads, d_kv, d_model))
            if rest == "self_attn.k_weight":
                k = hf_get(f"{hp}.layer.0.SelfAttention.k.weight")
                return fortran_order(k.reshape(n_heads, d_kv, d_model))
            if rest == "self_attn.v_weight":
                v = hf_get(f"{hp}.layer.0.SelfAttention.v.weight")
                return fortran_order(v.reshape(n_heads, d_kv, d_model))
            if rest == "self_attn.o_weight":
                o = hf_get(f"{hp}.layer.0.SelfAttention.o.weight")
                return fortran_order(o.reshape(d_model, n_heads, d_kv))

            if rest == "layer_norm_1.gamma":
                return fortran_order(
                    hf_get(f"{hp}.layer.1.layer_norm.weight"))
            if rest == "cross_attn.q_weight":
                q = hf_get(f"{hp}.layer.1.EncDecAttention.q.weight")
                return fortran_order(q.reshape(n_heads, d_kv, d_model))
            if rest == "cross_attn.k_weight":
                k = hf_get(f"{hp}.layer.1.EncDecAttention.k.weight")
                return fortran_order(k.reshape(n_heads, d_kv, d_model))
            if rest == "cross_attn.v_weight":
                v = hf_get(f"{hp}.layer.1.EncDecAttention.v.weight")
                return fortran_order(v.reshape(n_heads, d_kv, d_model))
            if rest == "cross_attn.o_weight":
                o = hf_get(f"{hp}.layer.1.EncDecAttention.o.weight")
                return fortran_order(o.reshape(d_model, n_heads, d_kv))

            if rest == "ff.layer_norm.gamma":
                return fortran_order(
                    hf_get(f"{hp}.layer.2.layer_norm.weight"))
            if rest == "ff.dense.gate_proj.weight":
                return fortran_order(
                    hf_get(f"{hp}.layer.2.DenseReluDense.wi.weight").T)
            if rest == "ff.dense.up_proj.weight":
                return fortran_order(
                    _get_ff_wi_2(hp, 2).T)
            if rest == "ff.dense.down_proj.weight":
                return fortran_order(
                    hf_get(f"{hp}.layer.2.DenseReluDense.wo.weight").T)

        raise ValueError(f"Unknown NNTile tensor: {name}")

    return convert


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert HF T5 checkpoint → NNTile format",
    )
    parser.add_argument(
        "--model", required=True,
        help="HF model (e.g. google/flan-t5-small) or local path",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory where NNTile files will be written",
    )
    parser.add_argument(
        "--encoder-prompt", default=None,
        help="Optional text for encoder input (source sequence)",
    )
    parser.add_argument(
        "--decoder-prompt", default=None,
        help="Optional text for decoder start (default: decoder_start_token)",
    )
    args = parser.parse_args()

    model_id = args.model

    if Path(model_id).is_dir():
        model_dir = Path(model_id)
    else:
        print(f"Downloading {model_id} to HF cache ...")
        model_dir = Path(snapshot_download(model_id))
        print(f"Snapshot: {model_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(str(model_dir))
    is_gated = getattr(config, "is_gated_act", True)
    if not is_gated:
        print(
            "ERROR: T5 v1.0 (non-gated) models are not supported. "
            "Use T5 v1.1+ gated models (e.g. google/flan-t5-small).",
            file=sys.stderr,
        )
        return 1
    print(f"Model: {config.model_type}  d_model={config.d_model}  "
          f"layers={config.num_layers}/{config.num_decoder_layers}  "
          f"vocab={config.vocab_size}")

    num_decoder_layers = getattr(
        config, "num_decoder_layers", config.num_layers)
    nntile_config = {
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "d_kv": config.d_kv,
        "d_ff": config.d_ff,
        "num_layers": config.num_layers,
        "num_decoder_layers": num_decoder_layers,
        "num_heads": config.num_heads,
        "layer_norm_epsilon": getattr(config, "layer_norm_epsilon", 1e-5),
        "eos_token_id": getattr(config, "eos_token_id", 1),
        "pad_token_id": getattr(config, "pad_token_id", 0),
        "decoder_start_token_id": getattr(config, "decoder_start_token_id", 0),
    }
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(nntile_config, indent=2) + "\n")
    print(f"Wrote {config_path}")

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"ERROR: no *.safetensors files found in {model_dir}",
              file=sys.stderr)
        return 1

    handles: list = []
    tensor_to_handle: dict[str, object] = {}
    for sf in st_files:
        h = safe_open(str(sf), framework="pt")
        handles.append(h)
        for key in h.keys():
            tensor_to_handle[key] = h

    tth = tensor_to_handle

    def hf_get(name: str) -> np.ndarray:
        return (tth[name].get_tensor(name).to(torch.float32).numpy())

    def has_tensor(name: str) -> bool:
        return name in tth

    specs = _output_specs(config)
    converter = _make_converter(config, hf_get, has_tensor)

    weights_path = out_dir / "weights.safetensors"
    print(f"Converting {len(specs)} tensors (streaming) ...")
    _write_safetensors_streaming(weights_path, specs, converter)
    print(f"Wrote {weights_path}")

    del handles, tensor_to_handle

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    if args.encoder_prompt is not None:
        enc_ids = tokenizer.encode(
            args.encoder_prompt,
            add_special_tokens=True,
            return_tensors="np",
        ).flatten()
        enc_str = ",".join(str(i) for i in enc_ids)
        (out_dir / "encoder_ids.txt").write_text(enc_str + "\n")
        print(f"Encoder prompt ({len(enc_ids)} tokens): {enc_ids.tolist()}")
        print(f"Wrote {out_dir / 'encoder_ids.txt'}")

    if args.decoder_prompt is not None:
        dec_ids = tokenizer.encode(
            args.decoder_prompt,
            add_special_tokens=False,
            return_tensors="np",
        ).flatten()
        dec_str = ",".join(str(i) for i in dec_ids)
        (out_dir / "decoder_ids.txt").write_text(dec_str + "\n")
        print(f"Decoder prompt ({len(dec_ids)} tokens): {dec_ids.tolist()}")
    else:
        dec_start = config.decoder_start_token_id
        if dec_start is None:
            dec_start = config.pad_token_id
        (out_dir / "decoder_ids.txt").write_text(str(dec_start) + "\n")
        print(f"Decoder start: {dec_start}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
