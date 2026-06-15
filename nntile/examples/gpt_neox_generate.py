#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file examples/gpt_neox_generate.py
# Convert a HuggingFace GPT-NeoX checkpoint to NNTile format for C++ inference.
#
# Usage:
#   python gpt_neox_generate.py \
#       --model EleutherAI/gpt-neox-125m \
#       --output-dir /tmp/nntile_gptneox \
#       --prompt "The meaning of life is"
#
# Then run the C++ binary:
#   ./gpt_neox_generate \
#       --config /tmp/nntile_gptneox/config.json \
#       --weights /tmp/nntile_gptneox/weights.safetensors \
#       --prompt-ids "$(cat /tmp/nntile_gptneox/prompt_ids.txt)" \
#       --max-tokens 32
#
# @version 1.1.0

"""Convert HF GPT-NeoX checkpoint to NNTile format and tokenize a prompt.

Produces:
  - weights.safetensors  NNTile-layout weight file
  - config.json  Model configuration readable by the C++ example
  - prompt_ids.txt  Comma-separated token IDs (if --prompt supplied)
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

from graph_bind import (
    as_bind_float32,
    gptneox_attn_o_weight,
    gptneox_attn_qkv_weight,
    linear_from_conv1d,
    linear_weight,
)


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
            err = f"{name}: expected {nbytes} bytes, got {len(raw)}"
            assert len(raw) == nbytes, err
            f.write(raw)
            del arr, raw


def _output_specs(config) -> list[tuple[str, tuple[int, ...]]]:
    """Return (nntile_name, shape) for every output tensor."""
    H = config.hidden_size
    V = config.vocab_size
    inter = config.intermediate_size
    nh = config.num_attention_heads
    hd = H // nh

    specs: list[tuple[str, tuple[int, ...]]] = []

    specs.append(("model.model.embed_tokens.vocab", (V, H)))
    specs.append(("model.model.norm.gamma", (H,)))
    specs.append(("model.model.norm.beta", (H,)))
    specs.append(("model.lm_head.weight", (V, H)))

    for i in range(config.num_hidden_layers):
        p = f"model.model.layers_{i}"
        specs.append((f"{p}.input_norm.gamma", (H,)))
        specs.append((f"{p}.input_norm.beta", (H,)))
        specs.append((f"{p}.post_attn_norm.gamma", (H,)))
        specs.append((f"{p}.post_attn_norm.beta", (H,)))

        specs.append((f"{p}.attention.q_weight", (H, hd, nh)))
        specs.append((f"{p}.attention.k_weight", (H, hd, nh)))
        specs.append((f"{p}.attention.v_weight", (H, hd, nh)))
        specs.append((f"{p}.attention.o_weight", (hd, nh, H)))

        specs.append((f"{p}.mlp.fc1.weight", (inter, H)))
        specs.append((f"{p}.mlp.fc1.bias", (inter,)))
        specs.append((f"{p}.mlp.fc2.weight", (H, inter)))
        specs.append((f"{p}.mlp.fc2.bias", (H,)))

    return specs


def _make_converter(
    config,
    hf_get: Callable[[str], np.ndarray],
    has_lm_head: bool,
) -> Callable[[str], np.ndarray]:
    """Return a function that converts a single NNTile tensor on demand."""
    H = config.hidden_size
    nh = config.num_attention_heads
    hd = H // nh

    def convert(name: str) -> np.ndarray:
        if name == "model.model.embed_tokens.vocab":
            return as_bind_float32(hf_get("gpt_neox.embed_in.weight"))

        if name == "model.model.norm.gamma":
            return as_bind_float32(hf_get("gpt_neox.final_layer_norm.weight"))
        if name == "model.model.norm.beta":
            return as_bind_float32(hf_get("gpt_neox.final_layer_norm.bias"))

        if name == "model.lm_head.weight":
            key = (
                "embed_out.weight"
                if has_lm_head
                else "gpt_neox.embed_in.weight"
            )
            return linear_weight(hf_get(key))

        parts = name.split(".")
        layer_idx = int(parts[2].split("_", 1)[1])
        rest = ".".join(parts[3:])
        hp = f"gpt_neox.layers.{layer_idx}"

        if rest == "input_norm.gamma":
            return as_bind_float32(hf_get(f"{hp}.input_layernorm.weight"))
        if rest == "input_norm.beta":
            return as_bind_float32(hf_get(f"{hp}.input_layernorm.bias"))
        if rest == "post_attn_norm.gamma":
            return as_bind_float32(
                hf_get(f"{hp}.post_attention_layernorm.weight"))
        if rest == "post_attn_norm.beta":
            return as_bind_float32(
                hf_get(f"{hp}.post_attention_layernorm.bias"))

        if rest == "attention.q_weight":
            qkv = hf_get(f"{hp}.attention.query_key_value.weight")
            qkv = qkv.reshape(nh, 3 * hd, H)
            return gptneox_attn_qkv_weight(qkv[:, :hd, :])

        if rest == "attention.k_weight":
            qkv = hf_get(f"{hp}.attention.query_key_value.weight")
            qkv = qkv.reshape(nh, 3 * hd, H)
            return gptneox_attn_qkv_weight(qkv[:, hd:2 * hd, :])

        if rest == "attention.v_weight":
            qkv = hf_get(f"{hp}.attention.query_key_value.weight")
            qkv = qkv.reshape(nh, 3 * hd, H)
            return gptneox_attn_qkv_weight(qkv[:, 2 * hd:3 * hd, :])

        if rest == "attention.o_weight":
            return gptneox_attn_o_weight(
                hf_get(f"{hp}.attention.dense.weight"), nh, hd, H)

        if rest == "mlp.fc1.weight":
            return linear_from_conv1d(
                hf_get(f"{hp}.mlp.dense_h_to_4h.weight"))
        if rest == "mlp.fc1.bias":
            return as_bind_float32(hf_get(f"{hp}.mlp.dense_h_to_4h.bias"))
        if rest == "mlp.fc2.weight":
            return linear_from_conv1d(
                hf_get(f"{hp}.mlp.dense_4h_to_h.weight"))
        if rest == "mlp.fc2.bias":
            return as_bind_float32(hf_get(f"{hp}.mlp.dense_4h_to_h.bias"))

        raise ValueError(f"Unknown NNTile tensor: {name}")

    return convert


def _load_tokenizer(*sources: str):
    """Try to load a tokenizer from multiple sources."""
    for src in sources:
        if not src:
            continue
        try:
            return AutoTokenizer.from_pretrained(src, use_fast=False)
        except Exception:
            pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert HF GPT-NeoX checkpoint → NNTile format",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF model name or path (e.g. EleutherAI/gpt-neox-125m)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where NNTile files will be written",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional text prompt to tokenize",
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
    print(
        f"Model: {config.model_type}  hidden={config.hidden_size}  "
        f"layers={config.num_hidden_layers}  vocab={config.vocab_size}"
    )

    nntile_config = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "layer_norm_eps": getattr(config, "layer_norm_eps", 1e-5),
        "rotary_pct": getattr(config, "rotary_pct", 0.25),
        "rotary_emb_base": getattr(config, "rotary_emb_base", 10000),
        "use_parallel_residual": getattr(
            config, "use_parallel_residual", True),
        "eos_token_id": getattr(config, "eos_token_id", 50256),
        "bos_token_id": getattr(config, "bos_token_id", 50256),
    }
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(nntile_config, indent=2) + "\n")
    print(f"Wrote {config_path}")

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(
            f"ERROR: no *.safetensors files found in {model_dir}",
            file=sys.stderr)
        return 1

    handles: list = []
    tensor_to_handle: dict[str, object] = {}
    for sf in st_files:
        h = safe_open(str(sf), framework="pt")
        handles.append(h)
        for key in h.keys():
            tensor_to_handle[key] = h

    n_hf = len(tensor_to_handle)
    print(f"Indexed {n_hf} tensors from {len(st_files)} shard(s) (mmap)")

    def hf_get(name: str, _m=tensor_to_handle) -> np.ndarray:
        return _m[name].get_tensor(name).to(torch.float32).numpy()

    specs = _output_specs(config)
    has_lm_head = "embed_out.weight" in tensor_to_handle
    converter = _make_converter(config, hf_get, has_lm_head)

    weights_path = out_dir / "weights.safetensors"
    print(f"Converting {len(specs)} tensors (streaming) ...")
    _write_safetensors_streaming(weights_path, specs, converter)
    print(f"Wrote {weights_path}")

    del handles, tensor_to_handle

    if args.prompt is not None:
        tokenizer = _load_tokenizer(str(model_dir), model_id)
        if tokenizer is None:
            print(
                "WARNING: tokenizer not loaded; skipping prompt tokenization.",
                file=sys.stderr,
            )
        else:
            ids = tokenizer.encode(args.prompt, add_special_tokens=True)
            ids_str = ",".join(str(i) for i in ids)
            ids_path = out_dir / "prompt_ids.txt"
            ids_path.write_text(ids_str + "\n")
            print(f"Prompt ({len(ids)} tokens): {ids}")
            print(f"Wrote {ids_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
