#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/bench_torch_native_middle_cpu_vs_nntile.py
# Middle-sized torch-native CPU vs nntile overhead benches (~1 min/train).

"""Run middle HF / CNN / DiT recipes on cpu and nntile; print markdown tables.

Uses :file:`torch_native_middle_recipes.json` (committed configs + steps /
batch / seq). Goal: show that StarPU / graph overhead becomes a smaller
fraction of wall time as model + batch grow — still on a single host core
(``ncpu=1``, ``torch.set_num_threads(1)``).

Example::

    python torch_nntile/examples/bench_torch_native_middle_cpu_vs_nntile.py \\
        --families hf,cnn,dit --markdown-out /tmp/middle_table.md
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


def _apply_protocol_env(protocol: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in protocol.get("env", {}).items():
        env[str(key)] = str(value)
    # Always pin host threads for this bench.
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"
    return env


def _build_cmd(
    *,
    here: Path,
    family: str,
    name: str,
    recipe: dict[str, Any],
    device: str,
    protocol: dict[str, Any],
    output_root: Path,
) -> list[str]:
    script = here / str(recipe["script"])
    config = here / str(recipe["config"])
    out_dir = output_root / family / f"{name}_{device}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ncpu = int(protocol.get("ncpu", 1))
    ncuda = int(protocol.get("ncuda", 0))
    seed = int(protocol.get("seed", 0))
    steps = int(recipe["steps"])
    batch_size = int(recipe["batch_size"])

    if recipe.get("is_gpt2"):
        seq_len = int(recipe["seq_len"])
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
            "--config",
            str(config),
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
        if device == "nntile" and ncuda == 0:
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
        "--config",
        str(config),
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
    if "seq_len" in recipe:
        cmd.extend(["--seq-len", str(int(recipe["seq_len"]))])
    if "dataset_split" in recipe:
        cmd.extend(["--dataset-split", str(recipe["dataset_split"])])
    if device == "nntile" and ncuda == 0:
        cmd.append("--restrict-cpu")
    return cmd


def run_one(
    *,
    here: Path,
    family: str,
    name: str,
    recipe: dict[str, Any],
    device: str,
    protocol: dict[str, Any],
    output_root: Path,
    env: dict[str, str],
) -> RunResult:
    cmd = _build_cmd(
        here=here,
        family=family,
        name=name,
        recipe=recipe,
        device=device,
        protocol=protocol,
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
        tail = (proc.stderr or proc.stdout or "")[-1000:].strip()
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
        "| Model | steps | batch | seq | CPU loss | nntile loss | "
        "CPU wall (s) | nntile wall (s) | nntile/CPU | Δ loss | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in order:
        row = by_name.get(name, {})
        cpu = row.get("cpu")
        nnt = row.get("nntile")
        if cpu is None or nnt is None:
            lines.append(
                f"| {name} | — | — | — | — | — | — | — | — | — | "
                "incomplete |"
            )
            continue
        recipe = cpu.recipe
        steps = recipe.get("steps", "—")
        batch = recipe.get("batch_size", "—")
        seq = recipe.get("seq_len", "—")
        if not cpu.ok or not nnt.ok:
            status = "FAIL"
            if not cpu.ok:
                status += " cpu"
            if not nnt.ok:
                status += " nntile"
            lines.append(
                f"| {name} | {steps} | {batch} | {seq} | "
                f"{cpu.loss if cpu.loss is not None else '—'} | "
                f"{nnt.loss if nnt.loss is not None else '—'} | "
                f"{(cpu.wall_s or float('nan')):.3f} | "
                f"{(nnt.wall_s or float('nan')):.3f} | — | — | "
                f"{status} |"
            )
            continue
        assert cpu.loss is not None and nnt.loss is not None
        assert cpu.wall_s is not None and nnt.wall_s is not None
        ratio = nnt.wall_s / max(cpu.wall_s, 1e-12)
        dloss = abs(cpu.loss - nnt.loss)
        lines.append(
            f"| {name} | {steps} | {batch} | {seq} | "
            f"{cpu.loss:.6f} | {nnt.loss:.6f} | "
            f"{cpu.wall_s:.3f} | {nnt.wall_s:.3f} | "
            f"{ratio:.2f}x | {dloss:.3e} | OK |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    default_recipes = here / "torch_native_middle_recipes.json"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--recipes",
        default=str(default_recipes),
        help="Path to torch_native_middle_recipes.json",
    )
    p.add_argument(
        "--families",
        default="hf,cnn,dit",
        help="Comma-separated: hf,cnn,dit",
    )
    p.add_argument(
        "--only",
        default="",
        help="Optional comma-separated model names to run",
    )
    p.add_argument(
        "--devices",
        default="cpu,nntile",
        help="Comma-separated devices (default: cpu,nntile)",
    )
    p.add_argument(
        "--output-root",
        default="/tmp/torch_native_middle_cpu_vs_nntile",
    )
    p.add_argument("--markdown-out", default="")
    p.add_argument(
        "--ncpu",
        type=int,
        default=None,
        help="Override recipe protocol ncpu",
    )
    p.add_argument(
        "--ncuda",
        type=int,
        default=None,
        help="Override recipe protocol ncuda",
    )
    args = p.parse_args(argv)

    recipes_doc = json.loads(Path(args.recipes).read_text(encoding="utf-8"))
    protocol = dict(recipes_doc.get("protocol", {}))
    if args.ncpu is not None:
        protocol["ncpu"] = args.ncpu
    if args.ncuda is not None:
        protocol["ncuda"] = args.ncuda

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    env = _apply_protocol_env(protocol)

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
                print(
                    f"==> {family}/{name} device={device} "
                    f"ncpu={protocol.get('ncpu')} "
                    f"ncuda={protocol.get('ncuda')}",
                    flush=True,
                )
                r = run_one(
                    here=here,
                    family=family,
                    name=name,
                    recipe=recipe,
                    device=device,
                    protocol=protocol,
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
        "# Middle torch-native CPU vs nntile\n\n"
        f"protocol ncpu={protocol.get('ncpu')} "
        f"ncuda={protocol.get('ncuda')} "
        f"seed={protocol.get('seed')} "
        f"host_threads={protocol.get('host_threads', 1)}\n"
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
