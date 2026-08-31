#!/usr/bin/env python3
"""Measure CUDA peak VRAM for Llama/T5 overhead configs (10-step train)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "torch_nntile" / "examples"


def run_train(script: str, config: Path, seq_len: int, gpu: int) -> float:
    out = tempfile.mkdtemp(prefix="vram_probe_")
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable,
        "-u",
        str(EXAMPLES / script),
        "train",
        "--device",
        "cuda",
        "--disable-tf32",
        "--seed",
        "42",
        "--no-shuffle",
        "--config",
        str(config),
        "--seq-len",
        str(seq_len),
        "--batch-size",
        "1",
        "--max-sequences",
        "10",
        "--epochs",
        "1",
        "--output-dir",
        out,
        "--no-checkpoint",
    ]
    subprocess.run(cmd, check=True, env=env)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    # Training runs in subprocess; re-measure via parsing is wrong. Re-run inline.
    del peak
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    import runpy

    sys.argv = cmd[2:]  # drop python -u
    runpy.run_path(str(EXAMPLES / script), run_name="__main__")
    return torch.cuda.max_memory_allocated() / (1024**3)


def measure_inline(script: str, config: Path, seq_len: int, gpu: int) -> float:
    import os
    import runpy

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = tempfile.mkdtemp(prefix="vram_probe_")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sys.argv = [
        str(EXAMPLES / script),
        "train",
        "--device",
        "cuda",
        "--disable-tf32",
        "--seed",
        "42",
        "--no-shuffle",
        "--config",
        str(config),
        "--seq-len",
        str(seq_len),
        "--batch-size",
        "1",
        "--max-sequences",
        "10",
        "--epochs",
        "1",
        "--output-dir",
        out,
        "--no-checkpoint",
    ]
    runpy.run_path(str(EXAMPLES / script), run_name="__main__")
    return torch.cuda.max_memory_allocated() / (1024**3)


LLAMA_SIZES = [
    ("xs", "llama_xs.json", 768),
    ("s", "llama_s.json", 1024),
    ("m", "llama_m.json", 1536),
    ("l", "llama_l.json", 2048),
    ("xl", "llama_xl.json", 2560),
]

T5_TEMPLATE = {
    "vocab_size": 2048,
    "d_kv": 64,
    "relative_attention_num_buckets": 32,
    "dropout_rate": 0.0,
    "layer_norm_epsilon": 1e-6,
    "feed_forward_proj": "relu",
    "is_encoder_decoder": True,
    "use_cache": False,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "decoder_start_token_id": 0,
}


def t5_config(
    *,
    d_model: int,
    d_ff: int,
    num_heads: int,
    max_dist: int,
    enc: int,
    dec: int,
) -> dict:
    cfg = dict(T5_TEMPLATE)
    cfg.update(
        {
            "d_model": d_model,
            "d_ff": d_ff,
            "num_heads": num_heads,
            "relative_attention_max_distance": max_dist,
            "num_layers": enc,
            "num_decoder_layers": dec,
            "d_kv": max(64, d_model // num_heads),
        }
    )
    return cfg


def try_t5(
    gpu: int,
    *,
    d_model: int,
    d_ff: int,
    num_heads: int,
    seq_len: int,
    enc: int,
    dec: int,
) -> float | None:
    import os
    import runpy

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = tempfile.mkdtemp(prefix="vram_t5_")
    cfg = t5_config(
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        max_dist=seq_len,
        enc=enc,
        dec=dec,
    )
    cfg_path = Path(out) / "cfg.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sys.argv = [
        str(EXAMPLES / "train_t5_hf_overhead.py"),
        "train",
        "--device",
        "cuda",
        "--disable-tf32",
        "--seed",
        "42",
        "--no-shuffle",
        "--config",
        str(cfg_path),
        "--seq-len",
        str(seq_len),
        "--batch-size",
        "1",
        "--max-sequences",
        "10",
        "--epochs",
        "1",
        "--output-dir",
        out + "/run",
        "--no-checkpoint",
    ]
    try:
        runpy.run_path(
            str(EXAMPLES / "train_t5_hf_overhead.py"), run_name="__main__"
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return None
        raise
    return torch.cuda.max_memory_allocated() / (1024**3)


def main() -> None:
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"GPU {gpu}", flush=True)
    llama_vram: dict[str, float] = {}
    for name, cfg, seq in LLAMA_SIZES:
        v = measure_inline("train_llama_hf.py", EXAMPLES / "overhead_llama" / cfg, seq, gpu)
        llama_vram[name] = v
        print(f"llama {name}: {v:.2f} GiB", flush=True)

    t5_specs = {
        "xs": (1536, 6144, 24, 768),
        "s": (2048, 8192, 16, 1024),
        "m": (3072, 12288, 24, 1536),
        "l": (4096, 16384, 32, 2048),
        "xl": (5760, 23040, 45, 2880),
    }
    for name, (dm, dff, heads, seq) in t5_specs.items():
        target = llama_vram[name]
        best = None
        for enc in range(1, 13):
            for dec in range(1, 13):
                v = try_t5(
                    gpu,
                    d_model=dm,
                    d_ff=dff,
                    num_heads=heads,
                    seq_len=seq,
                    enc=enc,
                    dec=dec,
                )
                if v is None:
                    continue
                err = abs(v - target)
                cand = (err, enc, dec, v)
                if best is None or cand < best:
                    best = cand
                print(
                    f"t5 {name} enc={enc} dec={dec}: {v:.2f} GiB "
                    f"(target {target:.2f}, err {err:.2f})",
                    flush=True,
                )
        if best:
            print(
                f"BEST t5 {name}: enc={best[1]} dec={best[2]} "
                f"{best[3]:.2f} GiB vs llama {target:.2f} GiB",
                flush=True,
            )


if __name__ == "__main__":
    main()
