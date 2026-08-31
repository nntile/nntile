#!/usr/bin/env python3
"""Find T5 enc/dec layer counts whose CUDA train VRAM matches Llama ladder."""

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
T5_DIR = EXAMPLES / "overhead_t5"

LLAMA_SIZES = {
    "xs": ("llama_xs.json", 768),
    "s": ("llama_s.json", 1024),
    "m": ("llama_m.json", 1536),
    "l": ("llama_l.json", 2048),
    "xl": ("llama_xl.json", 2560),
}

T5_WIDTH = {
    "xs": (1536, 6144, 24),
    "s": (2048, 8192, 16),
    "m": (3072, 12288, 24),
    "l": (4096, 16384, 32),
    "xl": (5760, 23040, 45),
}


def _train_cmd(script: str, config: Path, seq_len: int, out: str) -> list[str]:
    no_ckpt = (
        "--no-save-checkpoint"
        if "llama" in script
        else "--no-checkpoint"
    )
    return [
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
        no_ckpt,
    ]


def measure_peak_gib(
    script: str,
    config: Path,
    seq_len: int,
    *,
    gpu: int,
    env_extra: dict[str, str] | None = None,
) -> float | None:
    out = tempfile.mkdtemp(prefix="vram_peak_")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{EXAMPLES}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if env_extra:
        env.update(env_extra)
    probe = f"""
import runpy, sys, torch
sys.argv = {json.dumps(_train_cmd(script, config, seq_len, out)[2:])}
try:
    runpy.run_path({json.dumps(str(EXAMPLES / script))}, run_name='__main__')
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


def write_t5_config(
    path: Path,
    *,
    d_model: int,
    d_ff: int,
    num_heads: int,
    seq_len: int,
    enc: int,
    dec: int,
) -> None:
    d_kv = max(64, d_model // num_heads)
    cfg = {
        "_comment": (
            f"Overhead {path.stem.split('_', 1)[1].upper()}: "
            f"{enc} enc + {dec} dec layers, d_model={d_model}, "
            f"seq_len={seq_len} (CUDA VRAM matched to Llama)"
        ),
        "vocab_size": 2048,
        "d_model": d_model,
        "d_kv": d_kv,
        "d_ff": d_ff,
        "num_layers": enc,
        "num_decoder_layers": dec,
        "num_heads": num_heads,
        "relative_attention_num_buckets": 32,
        "relative_attention_max_distance": seq_len,
        "dropout_rate": 0.0,
        "layer_norm_epsilon": 1e-6,
        "feed_forward_proj": "relu",
        "is_encoder_decoder": True,
        "use_cache": False,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "decoder_start_token_id": 0,
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n")


def _try_t5_peak(
    tmp: Path,
    size: str,
    enc: int,
    dec: int,
    *,
    gpu: int,
) -> float | None:
    d_model, d_ff, num_heads = T5_WIDTH[size]
    _, seq_len = LLAMA_SIZES[size]
    cfg = tmp / f"enc{enc}_dec{dec}.json"
    write_t5_config(
        cfg,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        seq_len=seq_len,
        enc=enc,
        dec=dec,
    )
    return measure_peak_gib(
        "train_t5_hf_overhead.py",
        cfg,
        seq_len,
        gpu=gpu,
    )


def search_t5_layers(
    size: str,
    target_gib: float,
    *,
    gpu: int,
) -> tuple[int, int, float]:
    best: tuple[float, int, int, float] | None = None
    tmp = Path(tempfile.mkdtemp(prefix=f"t5_match_{size}_"))

    def consider(enc: int, dec: int, peak: float | None) -> None:
        nonlocal best
        if peak is None:
            return
        err = abs(peak - target_gib)
        cand = (err, enc, dec, peak)
        print(
            f"  {size} enc={enc} dec={dec}: {peak:.3f} GiB "
            f"(target {target_gib:.3f}, Δ={peak - target_gib:+.3f})",
            flush=True,
        )
        if best is None or cand < best:
            best = cand

    # Symmetric stacks first (fast sweep).
    for layers in range(1, 13):
        consider(layers, layers, _try_t5_peak(tmp, size, layers, layers, gpu=gpu))

    assert best is not None
    enc0, dec0 = best[1], best[2]
    # Refine in a small neighborhood around the best symmetric point.
    for enc in range(max(1, enc0 - 2), min(12, enc0 + 2) + 1):
        for dec in range(max(1, dec0 - 2), min(12, dec0 + 2) + 1):
            if enc == dec and abs(enc - enc0) <= 1:
                continue
            consider(enc, dec, _try_t5_peak(tmp, size, enc, dec, gpu=gpu))

    assert best is not None
    return best[1], best[2], best[3]


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
            seq_len,
            gpu=gpu,
        )
        assert peak is not None
        llama_targets[size] = peak
        print(f"llama {size}: {peak:.3f} GiB", flush=True)

    print("\n=== T5 layer search (CUDA peak VRAM) ===", flush=True)
    matches: dict[str, tuple[int, int, float]] = {}
    for size in LLAMA_SIZES:
        enc, dec, peak = search_t5_layers(
            size, llama_targets[size], gpu=gpu
        )
        matches[size] = (enc, dec, peak)
        print(
            f"MATCH {size}: enc={enc} dec={dec} -> {peak:.3f} GiB "
            f"(llama {llama_targets[size]:.3f} GiB, "
            f"Δ={peak - llama_targets[size]:+.3f})",
            flush=True,
        )

    if apply:
        print("\n=== Writing overhead_t5/*.json ===", flush=True)
        for size, (enc, dec, peak) in matches.items():
            d_model, d_ff, num_heads = T5_WIDTH[size]
            _, seq_len = LLAMA_SIZES[size]
            path = T5_DIR / f"t5_{size}.json"
            write_t5_config(
                path,
                d_model=d_model,
                d_ff=d_ff,
                num_heads=num_heads,
                seq_len=seq_len,
                enc=enc,
                dec=dec,
            )
            print(f"  wrote {path}  ({enc}+{dec}, {peak:.3f} GiB)", flush=True)

    summary = {
        "llama_gib": llama_targets,
        "t5_match": {
            k: {"enc": v[0], "dec": v[1], "gib": v[2]} for k, v in matches.items()
        },
    }
    out_path = REPO / "benchmark_logs" / "t5_vram_llama_match.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
