#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file examples/gptneo_generate.py
# Convert a HuggingFace GPT-Neo checkpoint to NNTile format for C++ inference.
#
# Usage:
#   python gptneo_generate.py \
#       --model EleutherAI/gpt-neo-1.3B \
#       --output-dir /tmp/nntile_gptneo \
#       --prompt "The capital of France is"
#
# Then run the C++ binary:
#   ./gptneo_generate \
#       --config /tmp/nntile_gptneo/config.json \
#       --weights /tmp/nntile_gptneo/weights.safetensors \
#       --prompt-ids "$(cat /tmp/nntile_gptneo/prompt_ids.txt)" \
#       --max-tokens 32
#
# @version 1.1.0

"""Convert HuggingFace GPT-Neo checkpoint to NNTile format and tokenize a prompt.

Produces:
  - weights.safetensors  NNTile-layout weight file
  - config.json  Model configuration readable by the C++ example
  - prompt_ids.txt  Comma-separated token IDs for the prompt (if --prompt supplied)
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


# ── Output tensor specs ───────────────────────────────────────────────────


def _output_specs(config) -> list[tuple[str, tuple[int, ...]]]:
    """Return (nntile_name, shape) for every output tensor."""
    H = config.hidden_size
    V = config.vocab_size
    inter = config.intermediate_size
    nh = config.num_attention_heads
    hd = H // nh

    specs: list[tuple[str, tuple[int, ...]]] = []

    specs.append(("model.model.wte.vocab", (H, V)))
    specs.append(("model.model.wpe.vocab", (H, config.max_position_embeddings)))
    specs.append(("model.model.norm.gamma", (H,)))
    specs.append(("model.lm_head.weight", (H, V)))

    for i in range(config.num_hidden_layers):
        p = f"model.model.layers_{i}"
        specs.append((f"{p}.input_norm.gamma", (H,)))
        specs.append((f"{p}.post_attn_norm.gamma", (H,)))

        specs.append((f"{p}.self_attn.q_weight", (nh, hd, H)))
        specs.append((f"{p}.self_attn.k_weight", (nh, hd, H)))
        specs.append((f"{p}.self_attn.v_weight", (nh, hd, H)))
        specs.append((f"{p}.self_attn.o_weight", (H, nh, hd)))
        specs.append((f"{p}.self_attn.o_bias", (H,)))

        specs.append((f"{p}.mlp.fc1.weight", (H, inter)))
        specs.append((f"{p}.mlp.fc2.weight", (inter, H)))

    return specs


# ── Per-tensor conversion ─────────────────────────────────────────────────


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
        if name == "model.model.wte.vocab":
            return fortran_order(hf_get("transformer.wte.weight").T)

        if name == "model.model.wpe.vocab":
            return fortran_order(hf_get("transformer.wpe.weight").T)

        if name == "model.model.norm.gamma":
            return fortran_order(hf_get("transformer.ln_f.weight"))

        if name == "model.lm_head.weight":
            key = ("lm_head.weight" if has_lm_head
                   else "transformer.wte.weight")
            return fortran_order(hf_get(key).T)

        parts = name.split(".")
        layer_idx = int(parts[2].split("_", 1)[1])
        rest = ".".join(parts[3:])
        hp = f"transformer.h.{layer_idx}"

        if rest == "input_norm.gamma":
            return fortran_order(hf_get(f"{hp}.ln_1.weight"))
        if rest == "post_attn_norm.gamma":
            return fortran_order(hf_get(f"{hp}.ln_2.weight"))

        if rest == "self_attn.q_weight":
            return fortran_order(
                hf_get(f"{hp}.attn.attention.q_proj.weight").reshape(nh, hd, H))
        if rest == "self_attn.k_weight":
            return fortran_order(
                hf_get(f"{hp}.attn.attention.k_proj.weight").reshape(nh, hd, H))
        if rest == "self_attn.v_weight":
            return fortran_order(
                hf_get(f"{hp}.attn.attention.v_proj.weight").reshape(nh, hd, H))
        if rest == "self_attn.o_weight":
            return fortran_order(
                hf_get(f"{hp}.attn.attention.c_proj.weight").reshape(H, nh, hd))
        if rest == "self_attn.o_bias":
            return fortran_order(hf_get(f"{hp}.attn.attention.c_proj.bias"))

        if rest == "mlp.fc1.weight":
            return fortran_order(hf_get(f"{hp}.mlp.c_fc.weight").T)
        if rest == "mlp.fc2.weight":
            return fortran_order(hf_get(f"{hp}.mlp.c_proj.weight").T)

        raise ValueError(f"Unknown NNTile tensor: {name}")

    return convert


# ── Tokenizer loading ─────────────────────────────────────────────────────


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


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert HF GPT-Neo checkpoint → NNTile format",
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model name (e.g. EleutherAI/gpt-neo-1.3B) or local path",
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
    inter = getattr(config, "intermediate_size", None) or 4 * config.hidden_size
    config.intermediate_size = inter

    print(f"Model: {config.model_type}  hidden={config.hidden_size}  "
          f"layers={config.num_layers}  vocab={config.vocab_size}")

    nntile_config = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_layers,
        "num_attention_heads": config.num_heads,
        "max_position_embeddings": config.max_position_embeddings,
        "layer_norm_eps": getattr(config, "layer_norm_epsilon", 1e-5),
        "eos_token_id": getattr(config, "eos_token_id", 50256),
        "bos_token_id": getattr(config, "bos_token_id", 50256),
        "window_size": getattr(config, "window_size", 256),
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

    print(f"Indexed {len(tensor_to_handle)} tensors from {len(st_files)} shard(s)")

    def hf_get(name: str, _m=tensor_to_handle) -> np.ndarray:
        return (_m[name]
                .get_tensor(name)
                .to(torch.float32)
                .numpy())

    specs = _output_specs(config)
    has_lm_head = "lm_head.weight" in tensor_to_handle
    converter = _make_converter(config, hf_get, has_lm_head)

    weights_path = out_dir / "weights.safetensors"
    print(f"Converting {len(specs)} tensors (streaming) ...")
    _write_safetensors_streaming(weights_path, specs, converter)
    print(f"Wrote {weights_path}")

    del handles, tensor_to_handle

    if args.prompt is not None:
        tokenizer = _load_tokenizer(str(model_dir), model_id)
        if tokenizer is None:
            print("WARNING: could not load tokenizer; skipping prompt tokenization.",
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
