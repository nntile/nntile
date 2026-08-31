#!/usr/bin/env python3
"""Find DiT layer counts whose CUDA train VRAM matches Llama ladder."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "torch_nntile" / "examples"
LLAMA_DIR = EXAMPLES / "overhead_llama"
DIT_DIR = EXAMPLES / "overhead_dit"

LLAMA_SIZES = {
    "xs": ("llama_xs.json", 768),
    "s": ("llama_s.json", 1024),
    "m": ("llama_m.json", 1536),
    "l": ("llama_l.json", 2048),
    "xl": ("llama_xl.json", 2560),
}

# hidden = heads * head_dim; sample_size so (sample_size/patch_size)^2 ~ seq_len
DIT_SHAPE = {
    "xs": (24, 64, 56),
    "s": (16, 128, 64),
    "m": (24, 128, 78),
    "l": (32, 128, 90),
    "xl": (45, 128, 108),
}


def _llama_train_cmd(config: Path, seq_len: int, out: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(EXAMPLES / "train_llama_hf.py"),
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
        "--no-save-checkpoint",
    ]


def _dit_train_cmd(config: Path, out: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(EXAMPLES / "train_dit_hf_overhead.py"),
        "train",
        "--device",
        "cuda",
        "--disable-tf32",
        "--seed",
        "42",
        "--no-shuffle",
        "--config",
        str(config),
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


def measure_peak_gib(
    script: str,
    config: Path,
    *,
    gpu: int,
    seq_len: int | None = None,
) -> float | None:
    out = tempfile.mkdtemp(prefix="vram_peak_")
    if script == "train_llama_hf.py":
        argv = _llama_train_cmd(config, int(seq_len), out)[2:]
        path = str(EXAMPLES / script)
    else:
        argv = _dit_train_cmd(config, out)[2:]
        path = str(EXAMPLES / script)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{EXAMPLES}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    probe = f"""
import runpy, sys, torch
sys.argv = {json.dumps(argv)}
try:
    runpy.run_path({json.dumps(path)}, run_name='__main__')
except SystemExit as exc:
    if exc.code not in (0, None):
        raise
print(f'peak_vram_gib={{torch.cuda.max_memory_allocated() / (1024**3):.4f}}', flush=True)
"""
    proc = subprocess.run(
        [sys.executable, "-u", "-c", probe],
        env=env,
        capture_output=True,
        text=True,
    )
    text = proc.stdout + proc.stderr
    if proc.returncode != 0:
        if "out of memory" in text.lower():
            return None
        raise RuntimeError(
            f"train failed ({script} {config.name}):\n{text[-4000:]}"
        )
    m = re.search(r"peak_vram_gib=([0-9.]+)", text)
    if not m:
        raise RuntimeError(f"no peak in output:\n{text[-4000:]}")
    return float(m.group(1))


def write_dit_config(
    path: Path,
    *,
    num_heads: int,
    head_dim: int,
    sample_size: int,
    num_layers: int,
    size: str,
    target_gib: float | None = None,
) -> None:
    hidden = num_heads * head_dim
    patches = (sample_size // 2) ** 2
    comment = (
        f"Overhead {size.upper()}: {num_layers} layers, hidden={hidden}, "
        f"sample_size={sample_size} ({patches} patches)"
    )
    if target_gib is not None:
        comment += f" (CUDA VRAM matched to Llama ~{target_gib:.2f} GiB)"
    cfg = {
        "_comment": comment,
        "sample_size": sample_size,
        "patch_size": 2,
        "in_channels": 3,
        "out_channels": 3,
        "num_layers": num_layers,
        "attention_head_dim": head_dim,
        "num_attention_heads": num_heads,
        "dropout": 0.0,
        "attention_bias": True,
        "activation_fn": "gelu-approximate",
        "num_embeds_ada_norm": 1000,
        "norm_type": "ada_norm_zero",
        "norm_elementwise_affine": False,
        "norm_eps": 1e-5,
        "upcast_attention": False,
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n")


def _try_dit_peak(
    tmp: Path,
    size: str,
    num_layers: int,
    *,
    gpu: int,
) -> float | None:
    num_heads, head_dim, sample_size = DIT_SHAPE[size]
    cfg = tmp / f"layers{num_layers}.json"
    write_dit_config(
        cfg,
        num_heads=num_heads,
        head_dim=head_dim,
        sample_size=sample_size,
        num_layers=num_layers,
        size=size,
    )
    return measure_peak_gib(
        "train_dit_hf_overhead.py",
        cfg,
        gpu=gpu,
    )


def search_dit_layers(
    size: str,
    target_gib: float,
    *,
    gpu: int,
) -> tuple[int, float]:
    cache: dict[int, float | None] = {}
    tmp = Path(tempfile.mkdtemp(prefix=f"dit_match_{size}_"))

    def peak(layers: int) -> float | None:
        if layers not in cache:
            cache[layers] = _try_dit_peak(tmp, size, layers, gpu=gpu)
            p = cache[layers]
            if p is not None:
                print(
                    f"  {size} layers={layers}: {p:.3f} GiB "
                    f"(target {target_gib:.3f}, Δ={p - target_gib:+.3f})",
                    flush=True,
                )
        return cache[layers]

    lo, hi = 1, 24
    while hi > lo and peak(hi) is None:
        hi -= 1
    if peak(lo) is None:
        raise RuntimeError(f"{size}: OOM at {lo} layers")

    # VRAM grows with depth; binary-search the largest layer count at/below target.
    while lo < hi:
        mid = (lo + hi + 1) // 2
        p = peak(mid)
        if p is None or p > target_gib:
            hi = mid - 1
        else:
            lo = mid

    candidates = sorted({max(1, lo - 1), lo, min(24, lo + 1)})
    best_layers = lo
    best_peak = peak(lo)
    assert best_peak is not None
    for layers in candidates:
        p = peak(layers)
        if p is None:
            continue
        if abs(p - target_gib) < abs(best_peak - target_gib):
            best_layers, best_peak = layers, p
    return best_layers, best_peak


def main() -> None:
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    apply = "--apply" in sys.argv
    print(f"GPU {gpu}  apply={apply}", flush=True)

    llama_targets: dict[str, float] = {}
    print("=== Llama CUDA peak VRAM (10-step train) ===", flush=True)
    for size, (cfg_name, seq_len) in LLAMA_SIZES.items():
        peak = measure_peak_gib(
            "train_llama_hf.py",
            LLAMA_DIR / cfg_name,
            gpu=gpu,
            seq_len=seq_len,
        )
        assert peak is not None
        llama_targets[size] = peak
        print(f"llama {size}: {peak:.3f} GiB", flush=True)

    print("\n=== DiT layer search (CUDA peak VRAM) ===", flush=True)
    matches: dict[str, tuple[int, float]] = {}
    for size in LLAMA_SIZES:
        layers, peak = search_dit_layers(
            size, llama_targets[size], gpu=gpu
        )
        matches[size] = (layers, peak)
        print(
            f"MATCH {size}: layers={layers} -> {peak:.3f} GiB "
            f"(llama {llama_targets[size]:.3f} GiB, "
            f"Δ={peak - llama_targets[size]:+.3f})",
            flush=True,
        )

    if apply:
        print("\n=== Writing overhead_dit/*.json ===", flush=True)
        for size, (layers, peak) in matches.items():
            num_heads, head_dim, sample_size = DIT_SHAPE[size]
            path = DIT_DIR / f"dit_{size}.json"
            write_dit_config(
                path,
                num_heads=num_heads,
                head_dim=head_dim,
                sample_size=sample_size,
                num_layers=layers,
                size=size,
                target_gib=llama_targets[size],
            )
            print(f"  wrote {path}  (layers={layers}, {peak:.3f} GiB)", flush=True)

    summary = {
        "llama_gib": llama_targets,
        "dit_match": {
            k: {"layers": v[0], "gib": v[1]} for k, v in matches.items()
        },
    }
    out_path = REPO / "benchmark_logs" / "dit_vram_llama_match.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
