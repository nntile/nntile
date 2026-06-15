#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file examples/gpt2_generate.py
# Convert a HuggingFace GPT-2 checkpoint to NNTile format for C++ inference.
#
# Usage:
#   python gpt2_generate.py \
#       --model gpt2 \
#       --output-dir /tmp/nntile_gpt2 \
#       --prompt "The capital of France is"
#
# Then run the C++ binary:
#   ./gpt2_generate \
#       --config /tmp/nntile_gpt2/config.json \
#       --weights /tmp/nntile_gpt2/weights.safetensors \
#       --prompt-ids "$(cat /tmp/nntile_gpt2/prompt_ids.txt)" \
#       --max-tokens 32
#
# @version 1.1.0

"""Convert HuggingFace GPT-2 checkpoint to NNTile format and tokenize a prompt.

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
    gpt2_attn_o_weight,
    gpt2_attn_qkv_weight,
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
            assert len(raw) == nbytes, (
                f"{name}: expected {nbytes} bytes, got {len(raw)}")
            f.write(raw)
            del arr, raw


def _conv1d_to_nntile_linear_weight(conv_weight: np.ndarray) -> np.ndarray:
    """Map HF GPT-2 Conv1D weight to graph ``Linear`` bind layout."""
    return linear_from_conv1d(conv_weight)


def _output_specs(config) -> list[tuple[str, tuple[int, ...]]]:
    """Return (nntile_name, shape) for every output tensor."""
    H = config.n_embd
    V = config.vocab_size
    n_layer = config.n_layer
    n_head = config.n_head
    n_inner = getattr(config, "n_inner", None) or 4 * H
    head_size = H // n_head

    specs: list[tuple[str, tuple[int, ...]]] = []

    specs.append(("model.transformer.wte.vocab", (V, H)))
    specs.append(("model.transformer.wpe.vocab", (config.n_positions, H)))

    for i in range(n_layer):
        p = f"model.transformer.h_{i}"
        specs.append((f"{p}.ln_1.gamma", (H,)))
        specs.append((f"{p}.ln_1.beta", (H,)))
        specs.append((f"{p}.ln_2.gamma", (H,)))
        specs.append((f"{p}.ln_2.beta", (H,)))

        specs.append((f"{p}.attn.q_weight", (H, head_size, n_head)))
        specs.append((f"{p}.attn.k_weight", (H, head_size, n_head)))
        specs.append((f"{p}.attn.v_weight", (H, head_size, n_head)))
        specs.append((f"{p}.attn.o_weight", (head_size, n_head, H)))
        specs.append((f"{p}.attn.q_bias", (n_head, head_size)))
        specs.append((f"{p}.attn.k_bias", (n_head, head_size)))
        specs.append((f"{p}.attn.v_bias", (n_head, head_size)))
        specs.append((f"{p}.attn.o_bias", (H,)))

        # MLP Linear weights: (out, in)
        specs.append((f"{p}.mlp.fc1.weight", (n_inner, H)))
        specs.append((f"{p}.mlp.fc1.bias", (n_inner,)))
        specs.append((f"{p}.mlp.fc2.weight", (H, n_inner)))
        specs.append((f"{p}.mlp.fc2.bias", (H,)))

    specs.append(("model.transformer.ln_f.gamma", (H,)))
    specs.append(("model.transformer.ln_f.beta", (H,)))
    specs.append(("model.lm_head.weight", (V, H)))

    return specs


def _make_converter(
    config,
    tensor_to_handle: dict,
) -> Callable[[str], np.ndarray]:
    """Return a function that converts a single NNTile tensor on demand."""
    H = config.n_embd
    n_head = config.n_head
    head_size = H // n_head
    tensor_keys = set(tensor_to_handle.keys())

    def hf_get(name: str) -> np.ndarray:
        h = tensor_to_handle[name]
        return h.get_tensor(name).to(torch.float32).numpy()

    def convert(name: str) -> np.ndarray:
        if name == "model.transformer.wte.vocab":
            return as_bind_float32(hf_get("wte.weight"))

        if name == "model.transformer.wpe.vocab":
            return as_bind_float32(hf_get("wpe.weight"))

        if name == "model.transformer.ln_f.gamma":
            return as_bind_float32(hf_get("ln_f.weight"))
        if name == "model.transformer.ln_f.beta":
            return as_bind_float32(hf_get("ln_f.bias"))

        if name == "model.lm_head.weight":
            if "lm_head.weight" in tensor_keys:
                return linear_weight(hf_get("lm_head.weight"))
            return linear_weight(hf_get("wte.weight"))

        parts = name.split(".")
        layer_idx = int(parts[2].split("_", 1)[1])
        rest = ".".join(parts[3:])
        hp = f"h.{layer_idx}"

        if rest == "ln_1.gamma":
            return as_bind_float32(hf_get(f"{hp}.ln_1.weight"))
        if rest == "ln_1.beta":
            return as_bind_float32(hf_get(f"{hp}.ln_1.bias"))
        if rest == "ln_2.gamma":
            return as_bind_float32(hf_get(f"{hp}.ln_2.weight"))
        if rest == "ln_2.beta":
            return as_bind_float32(hf_get(f"{hp}.ln_2.bias"))

        if rest == "attn.q_weight":
            c_attn = hf_get(f"{hp}.attn.c_attn.weight")
            return gpt2_attn_qkv_weight(c_attn[:, :H], H, n_head, head_size)
        if rest == "attn.k_weight":
            c_attn = hf_get(f"{hp}.attn.c_attn.weight")
            return gpt2_attn_qkv_weight(c_attn[:, H:2 * H], H, n_head, head_size)
        if rest == "attn.v_weight":
            c_attn = hf_get(f"{hp}.attn.c_attn.weight")
            return gpt2_attn_qkv_weight(
                c_attn[:, 2 * H:3 * H], H, n_head, head_size)
        if rest == "attn.o_weight":
            return gpt2_attn_o_weight(
                hf_get(f"{hp}.attn.c_proj.weight"), H, n_head, head_size)
        if rest == "attn.q_bias":
            c_bias = hf_get(f"{hp}.attn.c_attn.bias")
            return as_bind_float32(c_bias[:H].reshape(n_head, head_size))
        if rest == "attn.k_bias":
            c_bias = hf_get(f"{hp}.attn.c_attn.bias")
            return as_bind_float32(c_bias[H:2 * H].reshape(n_head, head_size))
        if rest == "attn.v_bias":
            c_bias = hf_get(f"{hp}.attn.c_attn.bias")
            return as_bind_float32(c_bias[2 * H:3 * H].reshape(n_head, head_size))
        if rest == "attn.o_bias":
            return as_bind_float32(hf_get(f"{hp}.attn.c_proj.bias"))

        if rest == "mlp.fc1.weight":
            return _conv1d_to_nntile_linear_weight(
                hf_get(f"{hp}.mlp.c_fc.weight"))
        if rest == "mlp.fc1.bias":
            return as_bind_float32(hf_get(f"{hp}.mlp.c_fc.bias"))
        if rest == "mlp.fc2.weight":
            return _conv1d_to_nntile_linear_weight(
                hf_get(f"{hp}.mlp.c_proj.weight"))
        if rest == "mlp.fc2.bias":
            return as_bind_float32(hf_get(f"{hp}.mlp.c_proj.bias"))

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
        description="Convert HF GPT-2 checkpoint → NNTile format",
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model name (e.g. gpt2) or local path",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory where NNTile files will be written",
    )
    parser.add_argument(
        "--prompt", default=None,
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
    print(f"Model: {config.model_type}  hidden={config.n_embd}  "
          f"layers={config.n_layer}  vocab={config.vocab_size}")

    n_inner = getattr(config, "n_inner", None) or 4 * config.n_embd
    nntile_config = {
        "vocab_size": config.vocab_size,
        "n_embd": config.n_embd,
        "n_inner": n_inner,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_positions": config.n_positions,
        "layer_norm_eps": getattr(config, "layer_norm_epsilon", 1e-5),
        "eos_token_id": getattr(config, "eos_token_id", 50256),
        "bos_token_id": getattr(config, "bos_token_id", 50256),
    }
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(nntile_config, indent=2) + "\n")
    print(f"Wrote {config_path}")

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(
            f"ERROR: no *.safetensors files in {model_dir}",
            file=sys.stderr)
        return 1

    handles: list = []
    tensor_to_handle: dict[str, object] = {}
    for sf in st_files:
        h = safe_open(str(sf), framework="pt")
        handles.append(h)
        for key in h.keys():
            tensor_to_handle[key] = h

    specs = _output_specs(config)
    converter = _make_converter(config, tensor_to_handle)

    weights_path = out_dir / "weights.safetensors"
    print(f"Converting {len(specs)} tensors (streaming) ...")
    _write_safetensors_streaming(weights_path, specs, converter)
    print(f"Wrote {weights_path}")

    del handles, tensor_to_handle

    if args.prompt is not None:
        tokenizer = _load_tokenizer(str(model_dir), model_id)
        if tokenizer is None:
            print(
                "WARNING: could not load tokenizer; skipping prompt.",
                file=sys.stderr)
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
