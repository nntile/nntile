#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file examples/llama_generate.py
# Convert a HuggingFace Llama checkpoint to NNTile format for C++ inference.
#
# Usage:
#   python llama_generate.py \
#       --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#       --output-dir /tmp/nntile_llama \
#       --prompt "The capital of France is"
#
# Then run the C++ binary:
#   ./llama_generate \
#       --config /tmp/nntile_llama/config.json \
#       --weights /tmp/nntile_llama/weights.safetensors \
#       --prompt-ids "$(cat /tmp/nntile_llama/prompt_ids.txt)" \
#       --max-tokens 32
#
# @version 1.1.0

"""Convert HuggingFace Llama checkpoint to NNTile format and tokenize a prompt.

Weights are converted in a streaming fashion: HF shard files are opened with
mmap, each tensor is read / converted / written one at a time, so peak memory
is proportional to the *largest single tensor* rather than the whole model.

Produces:
  - ``weights.safetensors``  NNTile-layout weight file (column-major, NNTile
    parameter naming).
  - ``config.json``  Model configuration readable by the C++ example.
  - ``prompt_ids.txt``  Comma-separated token IDs for the prompt (if
    ``--prompt`` is supplied).
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
    """Return C-contiguous array whose flat bytes equal NNTile column-major.

    ``safetensors`` writes C-order flat bytes.  NNTile reads those same bytes
    into Fortran-order (column-major) tiles.  Ravelling in F-order and
    reshaping in C-order produces a C-contiguous buffer that, when read
    linearly, gives the column-major element sequence NNTile expects.
    """
    a = np.asarray(arr, dtype=np.float32)
    return a.ravel("F").reshape(a.shape)


# ── Streaming safetensors writer ──────────────────────────────────────────


def _write_safetensors_streaming(
    path: str | Path,
    specs: Sequence[tuple[str, tuple[int, ...]]],
    get_data: Callable[[str], np.ndarray],
) -> None:
    """Write a safetensors file one tensor at a time.

    *specs* is a list of ``(name, shape)`` pairs (all float32).
    *get_data(name)* is called exactly once per tensor and must return a
    C-contiguous float32 ndarray with the matching shape.  The array (and
    its backing memory) can be freed as soon as the call returns.
    """
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


# ── Output tensor specs (shapes computed from config alone) ───────────────


def _output_specs(config) -> list[tuple[str, tuple[int, ...]]]:
    """Return ``(nntile_name, shape)`` for every output tensor."""
    H = config.hidden_size
    V = config.vocab_size
    inter = config.intermediate_size
    nh = config.num_attention_heads
    kv = getattr(config, "num_key_value_heads", nh)
    hd = H // nh
    gqa = kv < nh
    gs = nh // kv

    specs: list[tuple[str, tuple[int, ...]]] = []

    specs.append(("model.model.embed_tokens.vocab", (H, V)))
    specs.append(("model.model.norm.gamma", (H,)))
    specs.append(("model.lm_head.weight", (H, V)))

    for i in range(config.num_hidden_layers):
        p = f"model.model.layers_{i}"
        specs.append((f"{p}.input_norm.gamma", (H,)))
        specs.append((f"{p}.post_attn_norm.gamma", (H,)))

        if gqa:
            specs.append((f"{p}.attention.q_weight", (gs, kv, hd, H)))
            specs.append((f"{p}.attention.o_weight", (H, gs, kv, hd)))
        else:
            specs.append((f"{p}.attention.q_weight", (nh, hd, H)))
            specs.append((f"{p}.attention.o_weight", (H, nh, hd)))

        specs.append((f"{p}.attention.k_weight", (kv, hd, H)))
        specs.append((f"{p}.attention.v_weight", (kv, hd, H)))

        specs.append((f"{p}.mlp.gate_proj.weight", (H, inter)))
        specs.append((f"{p}.mlp.up_proj.weight", (H, inter)))
        specs.append((f"{p}.mlp.down_proj.weight", (inter, H)))

    return specs


# ── Per-tensor conversion (loads one HF tensor at a time) ─────────────────


def _make_converter(
    config,
    hf_get: Callable[[str], np.ndarray],
    has_lm_head: bool,
) -> Callable[[str], np.ndarray]:
    """Return a function that converts a single NNTile tensor on demand."""
    H = config.hidden_size
    nh = config.num_attention_heads
    kv = getattr(config, "num_key_value_heads", nh)
    hd = H // nh
    gqa = kv < nh
    gs = nh // kv

    def convert(name: str) -> np.ndarray:
        if name == "model.model.embed_tokens.vocab":
            return fortran_order(hf_get("model.embed_tokens.weight").T)

        if name == "model.model.norm.gamma":
            return fortran_order(hf_get("model.norm.weight"))

        if name == "model.lm_head.weight":
            key = ("lm_head.weight" if has_lm_head
                   else "model.embed_tokens.weight")
            return fortran_order(hf_get(key).T)

        # Layer tensors: model.model.layers_{i}.<rest>
        parts = name.split(".")
        layer_idx = int(parts[2].split("_", 1)[1])
        rest = ".".join(parts[3:])
        hp = f"model.layers.{layer_idx}"

        if rest == "input_norm.gamma":
            return fortran_order(hf_get(f"{hp}.input_layernorm.weight"))
        if rest == "post_attn_norm.gamma":
            return fortran_order(
                hf_get(f"{hp}.post_attention_layernorm.weight"))

        if rest == "attention.q_weight":
            q = hf_get(f"{hp}.self_attn.q_proj.weight")
            if gqa:
                return fortran_order(
                    q.reshape(kv, gs, hd, H).transpose(1, 0, 2, 3))
            return fortran_order(q.reshape(nh, hd, H))

        if rest == "attention.k_weight":
            return fortran_order(
                hf_get(f"{hp}.self_attn.k_proj.weight").reshape(kv, hd, H))

        if rest == "attention.v_weight":
            return fortran_order(
                hf_get(f"{hp}.self_attn.v_proj.weight").reshape(kv, hd, H))

        if rest == "attention.o_weight":
            o = hf_get(f"{hp}.self_attn.o_proj.weight")
            if gqa:
                return fortran_order(
                    o.reshape(H, kv, gs, hd).transpose(0, 2, 1, 3))
            return fortran_order(o.reshape(H, nh, hd))

        if rest == "mlp.gate_proj.weight":
            return fortran_order(hf_get(f"{hp}.mlp.gate_proj.weight").T)
        if rest == "mlp.up_proj.weight":
            return fortran_order(hf_get(f"{hp}.mlp.up_proj.weight").T)
        if rest == "mlp.down_proj.weight":
            return fortran_order(hf_get(f"{hp}.mlp.down_proj.weight").T)

        raise ValueError(f"Unknown NNTile tensor: {name}")

    return convert


# ── Tokenizer loading (with fallbacks) ────────────────────────────────────


def _load_tokenizer(*sources: str):
    """Try to load a tokenizer from multiple sources with fallbacks."""
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
        description="Convert HF Llama checkpoint → NNTile format",
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model name "
             "(e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
             " or local path to a directory with "
             "config.json + *.safetensors.  Remote names"
             " are downloaded to the HF cache.",
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

    # If it looks like a local path, use it directly; otherwise fetch from hub.
    if Path(model_id).is_dir():
        model_dir = Path(model_id)
    else:
        print(f"Downloading {model_id} to HF cache ...")
        model_dir = Path(snapshot_download(model_id))
        print(f"Snapshot: {model_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load HF config ────────────────────────────────────────────────
    config = AutoConfig.from_pretrained(str(model_dir))
    print(f"Model: {config.model_type}  hidden={config.hidden_size}  "
          f"layers={config.num_hidden_layers}  vocab={config.vocab_size}")

    # ── Write NNTile config.json ──────────────────────────────────────
    kv_heads = getattr(config, "num_key_value_heads",
                       config.num_attention_heads)
    nntile_config = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": kv_heads,
        "rms_norm_eps": getattr(config, "rms_norm_eps", 1e-6),
        "eos_token_id": getattr(config, "eos_token_id", 2),
        "bos_token_id": getattr(config, "bos_token_id", 1),
    }
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(nntile_config, indent=2) + "\n")
    print(f"Wrote {config_path}")

    # ── Open HF shard files (mmap, lazy tensor access) ────────────────
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

    n_hf = len(tensor_to_handle)
    print(f"Indexed {n_hf} tensors from {len(st_files)} shard(s) (mmap)")

    def hf_get(name: str, _m=tensor_to_handle) -> np.ndarray:
        return (_m[name]
                .get_tensor(name)
                .to(torch.float32)
                .numpy())

    # ── Stream-convert weights ────────────────────────────────────────
    specs = _output_specs(config)
    has_lm_head = "lm_head.weight" in tensor_to_handle
    converter = _make_converter(config, hf_get, has_lm_head)

    weights_path = out_dir / "weights.safetensors"
    print(f"Converting {len(specs)} tensors (streaming) ...")
    _write_safetensors_streaming(weights_path, specs, converter)
    print(f"Wrote {weights_path}")

    del handles, tensor_to_handle

    # ── Tokenize prompt (optional) ────────────────────────────────────
    if args.prompt is not None:
        tokenizer = _load_tokenizer(str(model_dir), model_id)
        if tokenizer is None:
            print("WARNING: could not load tokenizer; skipping prompt "
                  "tokenization.\n  Supply --prompt-ids to the C++ binary "
                  "directly.", file=sys.stderr)
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
