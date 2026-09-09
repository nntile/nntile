#!/usr/bin/env python3
"""Probe HF(nntile) CNN overhead configs for on-device VRAM (D2H=0).

Runs a 10-step overlap train on ``device=nntile`` and reports nvidia-smi
peak plus StarPU bus D2H. Exit 0 if the run finished with D2H ~ 0;
exit 2 if it paged or OOM'd.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "torch_nntile" / "examples" / "train_cnn_hf_overhead.py"
FAMILIES = (
    "lenet",
    "resnet",
    "vgg",
    "mobilenet",
    "unet",
    "unet_modern",
)
SIZES = ("xs", "s", "m", "l", "xl")
# StarPU prints a dummy CUDA→NUMA transfer of 0 bytes; treat above this
# as paging.
D2H_PAGE_GB = 0.01


def config_path(family: str, size: str) -> Path:
    return (
        REPO
        / "torch_nntile"
        / "examples"
        / f"overhead_{family}"
        / f"{family}_{size}.json"
    )


def parse_bus_gb(text: str) -> tuple[float | None, float | None]:
    h2d = None
    d2h = None
    m = re.search(
        r"NUMA 0\s*->\s*CUDA 0\s+([0-9.]+)\s+GB",
        text,
    )
    if m:
        h2d = float(m.group(1))
    m = re.search(
        r"CUDA 0\s*->\s*NUMA 0\s+([0-9.]+)\s+GB",
        text,
    )
    if m:
        d2h = float(m.group(1))
    return h2d, d2h


def parse_peak(text: str) -> float | None:
    m = re.search(r"peak_vram_gib=([0-9.]+)", text)
    return float(m.group(1)) if m else None


def parse_loss(text: str) -> float | None:
    m = re.search(r"\[nntile\] final loss=([0-9.]+)", text)
    return float(m.group(1)) if m else None


def is_oom(text: str) -> bool:
    low = text.lower()
    return any(
        s in low
        for s in (
            "out of memory",
            "cuda oom",
            "starpu memory",
            "cannot allocate",
        )
    )


def run_probe(
    *,
    family: str,
    size: str,
    gpu: str,
    logdir: Path,
    steps: int,
) -> dict:
    cfg = config_path(family, size)
    if not cfg.is_file():
        raise FileNotFoundError(f"missing config {cfg}")
    tag = f"{family}_{size}_nntile_vram"
    out = logdir / tag
    out.mkdir(parents=True, exist_ok=True)
    log_path = logdir / f"{tag}.log"
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env["CUDA_VISIBLE_DEVICES"] = gpu
    examples_path = str(REPO / "torch_nntile" / "examples")
    extra_pp = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{extra_pp}:{REPO / 'torch_nntile'}:{examples_path}"
        if extra_pp
        else f"{REPO / 'torch_nntile'}:{examples_path}"
    )
    env.setdefault("STARPU_LIMIT_CUDA_MEM", "46000")
    env.setdefault("STARPU_BUS_STATS", "1")
    env.setdefault("STARPU_SILENT", "1")
    env.setdefault("STARPU_FXT_TRACE", "0")
    env.setdefault("STARPU_WORKERS_NOBIND", "1")
    cmd = [
        sys.executable,
        "-u",
        str(TRAIN),
        "train",
        "--model",
        family,
        "--device",
        "nntile",
        "--restrict-cuda",
        "--ncpu",
        "0",
        "--ncuda",
        "1",
        "--seed",
        "42",
        "--no-shuffle",
        "--config",
        str(cfg),
        "--batch-size",
        "1",
        "--max-sequences",
        str(steps),
        "--epochs",
        "1",
        "--output-dir",
        str(out),
        "--no-checkpoint",
        "--verbose",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    text = proc.stdout + "\n" + proc.stderr
    log_path.write_text(text, encoding="utf-8")
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    h2d, d2h = parse_bus_gb(text)
    peak = parse_peak(text)
    loss = parse_loss(text)
    oom = is_oom(text) or proc.returncode != 0
    paged = d2h is not None and d2h >= D2H_PAGE_GB
    fit = (not oom) and (d2h is not None) and (not paged)
    # If bus stats missing but the run succeeded and peak is well under
    # the card, still treat as unknown rather than a fit.
    status = "fit" if fit else ("oom" if oom else ("page" if paged else "unknown"))
    return {
        "family": family,
        "size": size,
        "status": status,
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "peak_vram_gib": peak,
        "h2d_gb": h2d,
        "d2h_gb": d2h,
        "loss": loss,
        "log": str(log_path),
    }


def format_row(row: dict) -> str:
    peak = row["peak_vram_gib"]
    h2d = row["h2d_gb"]
    d2h = row["d2h_gb"]
    peak_s = f"{peak:.1f} GiB" if peak is not None else "?"
    h2d_s = f"{h2d:.2f} GB" if h2d is not None else "?"
    d2h_s = f"{d2h:.4f} GB" if d2h is not None else "?"
    return (
        f"{row['family']:12s} {row['size']:3s}  {row['status']:8s}  "
        f"peak={peak_s:10s}  H2D={h2d_s:10s}  D2H={d2h_s:12s}  "
        f"wall={row['elapsed_s']:.1f}s  rc={row['returncode']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=list(FAMILIES))
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=list(SIZES),
        default=list(SIZES),
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    args.logdir.mkdir(parents=True, exist_ok=True)
    worst = 0
    for size in args.sizes:
        print(
            f"probe {args.family} {size} gpu={args.gpu} steps={args.steps}",
            flush=True,
        )
        row = run_probe(
            family=args.family,
            size=size,
            gpu=args.gpu,
            logdir=args.logdir,
            steps=args.steps,
        )
        print(format_row(row), flush=True)
        if row["status"] != "fit":
            worst = 2
            print(f"  log: {row['log']}", flush=True)
            tail = Path(row["log"]).read_text(encoding="utf-8")[-1500:]
            print(tail, flush=True)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
