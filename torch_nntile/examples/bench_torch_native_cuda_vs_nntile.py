#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/bench_torch_native_cuda_vs_nntile.py
# Tiny + middle torch-native CUDA vs nntile (single GPU) benches.

"""Compare ``--device cuda`` vs ``--device nntile`` (``ncuda=1``).

Tiny defaults match the CPU showcase smokes; middle uses
``torch_native_middle_recipes.json``.

Example::

    export CUDA_VISIBLE_DEVICES=1
    export LD_LIBRARY_PATH=$PWD/install/lib:/opt/conda/envs/nntile/lib:$LD_LIBRARY_PATH
    python torch_nntile/examples/bench_torch_native_cuda_vs_nntile.py \\
        --suite tiny --families hf,cnn,dit \\
        --markdown-out /tmp/torch_native_tiny_cuda.md
    python torch_nntile/examples/bench_torch_native_cuda_vs_nntile.py \\
        --suite middle --families hf,cnn,dit \\
        --markdown-out /tmp/torch_native_middle_cuda.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOSS_RE = re.compile(r"loss=([0-9.]+)")
WALL_RE = re.compile(r"wall=([0-9.]+)s")
GPT2_WALL_RE = re.compile(
    r"timing (?:nntile|torch) train wall[^:]*:\s*([0-9.]+)s"
)

TINY_RECIPES: dict[str, dict[str, dict[str, Any]]] = {
    "hf": {
        "gpt2": {
            "script": "train_gpt2_hf.py",
            "is_gpt2": True,
            "steps": 1,
            "seq_len": 16,
            "batch_size": 1,
        },
        "gpt-neo": {
            "script": "train_gpt_neo_hf.py",
            "steps": 1,
            "seq_len": 16,
            "batch_size": 1,
        },
        "gpt-neox": {
            "script": "train_gpt_neox_hf.py",
            "steps": 1,
            "seq_len": 16,
            "batch_size": 1,
        },
        "llama": {
            "script": "train_llama_hf.py",
            "steps": 1,
            "seq_len": 16,
            "batch_size": 1,
        },
        "bert": {
            "script": "train_bert_hf.py",
            "steps": 1,
            "seq_len": 16,
            "batch_size": 1,
        },
        "roberta": {
            "script": "train_roberta_hf.py",
            "steps": 1,
            "seq_len": 16,
            "batch_size": 1,
        },
        "t5": {
            "script": "train_t5_hf.py",
            "steps": 1,
            "seq_len": 16,
            "batch_size": 1,
        },
    },
    "cnn": {
        "lenet": {
            "script": "train_lenet_tiny.py",
            "steps": 1,
            "batch_size": 2,
        },
        "resnet": {
            "script": "train_resnet_tiny.py",
            "steps": 1,
            "batch_size": 2,
        },
        "vgg": {
            "script": "train_vgg_tiny.py",
            "steps": 1,
            "batch_size": 2,
        },
        "mobilenet": {
            "script": "train_mobilenet_tiny.py",
            "steps": 1,
            "batch_size": 2,
        },
        "unet": {
            "script": "train_unet_tiny.py",
            "steps": 1,
            "batch_size": 2,
        },
        "unet_modern": {
            "script": "train_unet_modern_tiny.py",
            "steps": 1,
            "batch_size": 2,
        },
    },
    "dit": {
        "dit": {
            "script": "train_dit_hf.py",
            "steps": 1,
            "batch_size": 2,
        },
    },
}


@dataclass
class RunResult:
    family: str
    name: str
    device: str
    ok: bool
    loss: float | None
    wall_s: float | None
    elapsed_s: float
    stderr_tail: str
    recipe: dict[str, Any]


def _host_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"
    env.setdefault("STARPU_SILENT", "1")
    env.setdefault("STARPU_FXT_TRACE", "0")
    env.setdefault("STARPU_WORKERS_NOBIND", "1")
    return env


def _build_cmd(
    *,
    here: Path,
    family: str,
    name: str,
    recipe: dict[str, Any],
    device: str,
    ncpu: int,
    ncuda: int,
    seed: int,
    output_root: Path,
) -> list[str]:
    script = here / str(recipe["script"])
    out_dir = output_root / family / f"{name}_{device}"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = int(recipe["steps"])
    batch_size = int(recipe["batch_size"])

    if recipe.get("is_gpt2"):
        seq_len = int(recipe.get("seq_len", 16))
        cmd = [
            sys.executable,
            str(script),
            "train",
            "--device",
            device,
            "--seed",
            str(seed),
            "--data-seed",
            str(seed),
            "--epochs",
            "1",
            "--seq-len",
            str(seq_len),
            "--batch-size",
            str(batch_size),
            "--max-sequences",
            str(max(batch_size, steps * batch_size)),
            "--ncpu",
            str(ncpu),
            "--ncuda",
            str(ncuda),
            "--output-dir",
            str(out_dir),
            "--no-shuffle",
        ]
        if "config" in recipe:
            cmd.extend(["--config", str(here / str(recipe["config"]))])
        if device == "cuda":
            cmd.append("--disable-tf32")
        if device == "nntile" and ncuda > 0:
            cmd.append("--restrict-cuda")
        elif device == "nntile":
            cmd.append("--restrict-cpu")
        return cmd

    cmd = [
        sys.executable,
        str(script),
        "train",
        "--device",
        device,
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--ncpu",
        str(ncpu),
        "--ncuda",
        str(ncuda),
        "--output-dir",
        str(out_dir),
    ]
    if "config" in recipe:
        cmd.extend(["--config", str(here / str(recipe["config"]))])
    if "seq_len" in recipe:
        cmd.extend(["--seq-len", str(int(recipe["seq_len"]))])
    if "dataset_split" in recipe:
        cmd.extend(["--dataset-split", str(recipe["dataset_split"])])
    if device == "nntile" and ncuda > 0:
        cmd.append("--restrict-cuda")
    elif device == "nntile":
        cmd.append("--restrict-cpu")
    return cmd


def run_one(
    *,
    here: Path,
    family: str,
    name: str,
    recipe: dict[str, Any],
    device: str,
    ncpu: int,
    ncuda: int,
    seed: int,
    output_root: Path,
    env: dict[str, str],
) -> RunResult:
    cmd = _build_cmd(
        here=here,
        family=family,
        name=name,
        recipe=recipe,
        device=device,
        ncpu=ncpu,
        ncuda=ncuda,
        seed=seed,
        output_root=output_root,
    )
    print("    " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(here.parent.parent),
        env=env,
    )
    elapsed = time.perf_counter() - t0
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    losses = LOSS_RE.findall(text)
    loss = float(losses[-1]) if losses else None
    wall = None
    if recipe.get("is_gpt2"):
        m = GPT2_WALL_RE.search(text)
        if m:
            wall = float(m.group(1))
    if wall is None:
        m = WALL_RE.search(text)
        if m:
            wall = float(m.group(1))
    ok = proc.returncode == 0 and loss is not None
    tail = ""
    if not ok:
        tail = (proc.stderr or proc.stdout or "")[-1200:].strip()
    return RunResult(
        family=family,
        name=name,
        device=device,
        ok=ok,
        loss=loss,
        wall_s=wall,
        elapsed_s=elapsed,
        stderr_tail=tail,
        recipe=recipe,
    )


def format_family_table(
    family: str,
    results: list[RunResult],
    order: list[str],
) -> str:
    by_name: dict[str, dict[str, RunResult]] = {}
    for r in results:
        if r.family != family:
            continue
        by_name.setdefault(r.name, {})[r.device] = r

    lines = [
        f"### {family}",
        "",
        "| Model | steps | batch | seq | CUDA loss | nntile loss | "
        "CUDA wall (s) | nntile wall (s) | nntile/CUDA | Δ loss | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in order:
        row = by_name.get(name, {})
        cuda = row.get("cuda")
        nnt = row.get("nntile")
        if cuda is None or nnt is None:
            lines.append(
                f"| {name} | — | — | — | — | — | — | — | — | — | "
                "incomplete |"
            )
            continue
        recipe = cuda.recipe
        steps = recipe.get("steps", "—")
        batch = recipe.get("batch_size", "—")
        seq = recipe.get("seq_len", "—")
        if not cuda.ok or not nnt.ok:
            status = "FAIL"
            if not cuda.ok:
                status += " cuda"
            if not nnt.ok:
                status += " nntile"
            lines.append(
                f"| {name} | {steps} | {batch} | {seq} | "
                f"{cuda.loss if cuda.loss is not None else '—'} | "
                f"{nnt.loss if nnt.loss is not None else '—'} | "
                f"{(cuda.wall_s or float('nan')):.3f} | "
                f"{(nnt.wall_s or float('nan')):.3f} | — | — | "
                f"{status} |"
            )
            continue
        assert cuda.loss is not None and nnt.loss is not None
        assert cuda.wall_s is not None and nnt.wall_s is not None
        ratio = nnt.wall_s / max(cuda.wall_s, 1e-12)
        dloss = abs(cuda.loss - nnt.loss)
        lines.append(
            f"| {name} | {steps} | {batch} | {seq} | "
            f"{cuda.loss:.6f} | {nnt.loss:.6f} | "
            f"{cuda.wall_s:.3f} | {nnt.wall_s:.3f} | "
            f"{ratio:.2f}x | {dloss:.3e} | OK |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    default_middle = here / "torch_native_middle_recipes.json"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--suite",
        choices=("tiny", "middle"),
        default="tiny",
        help="tiny showcase smokes or middle recipes",
    )
    p.add_argument(
        "--recipes",
        default=str(default_middle),
        help="Middle recipes JSON (ignored for --suite tiny)",
    )
    p.add_argument("--families", default="hf,cnn,dit")
    p.add_argument("--only", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ncpu", type=int, default=0)
    p.add_argument("--ncuda", type=int, default=1)
    p.add_argument(
        "--output-root",
        default="/tmp/torch_native_cuda_vs_nntile",
    )
    p.add_argument("--markdown-out", default="")
    args = p.parse_args(argv)

    if args.suite == "tiny":
        recipes_doc: dict[str, Any] = dict(TINY_RECIPES)
    else:
        recipes_doc = json.loads(
            Path(args.recipes).read_text(encoding="utf-8")
        )

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    output_root = Path(args.output_root) / args.suite
    output_root.mkdir(parents=True, exist_ok=True)
    env = _host_env()
    devices = ("cuda", "nntile")

    results: list[RunResult] = []
    sections: list[str] = []
    for family in families:
        group = recipes_doc.get(family, {})
        if not isinstance(group, dict) or not group:
            print(f"WARNING: no recipes for family={family}", flush=True)
            continue
        order = list(group.keys())
        for name in order:
            if only and name not in only:
                continue
            recipe = dict(group[name])
            for device in devices:
                ncpu = args.ncpu if device == "nntile" else 0
                ncuda = args.ncuda if device == "nntile" else 0
                print(
                    f"==> {args.suite}/{family}/{name} device={device} "
                    f"ncpu={ncpu} ncuda={ncuda}",
                    flush=True,
                )
                r = run_one(
                    here=here,
                    family=family,
                    name=name,
                    recipe=recipe,
                    device=device,
                    ncpu=ncpu,
                    ncuda=ncuda,
                    seed=args.seed,
                    output_root=output_root,
                    env=env,
                )
                status = "OK" if r.ok else "FAIL"
                print(
                    f"    {status} loss={r.loss} wall={r.wall_s} "
                    f"elapsed={r.elapsed_s:.3f}s",
                    flush=True,
                )
                if not r.ok and r.stderr_tail:
                    print(f"    stderr_tail:\n{r.stderr_tail}", flush=True)
                results.append(r)
        sections.append(format_family_table(family, results, order))

    header = (
        f"# {args.suite.capitalize()} torch-native CUDA vs nntile\n\n"
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
        f"ncpu={args.ncpu} ncuda={args.ncuda} seed={args.seed}\n"
    )
    table = header + "\n\n".join(sections) + "\n"
    print("\n" + table)
    if args.markdown_out:
        Path(args.markdown_out).write_text(table, encoding="utf-8")
        print(f"Wrote {args.markdown_out}")

    failed = sum(1 for r in results if not r.ok)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
