#!/usr/bin/env python3
"""Fill docs/dev/<family>_hf_overhead_scale.md from a CNN ladder summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SIZES = ("xs", "s", "m", "l", "xl")
SPATIAL = {
    "lenet": {
        "xs": "32²",
        "s": "48²",
        "m": "64²",
        "l": "80²",
        "xl": "112²",
    },
    "resnet": {
        "xs": "32²",
        "s": "48²",
        "m": "64²",
        "l": "80²",
        "xl": "192²",
    },
    "vgg": {
        "xs": "32²",
        "s": "48²",
        "m": "64²",
        "l": "80²",
        "xl": "144²",
    },
    "mobilenet": {
        "xs": "32²",
        "s": "48²",
        "m": "64²",
        "l": "80²",
        "xl": "128²",
    },
    "unet": {
        "xs": "32²",
        "s": "48²",
        "m": "64²",
        "l": "80²",
        "xl": "192²",
    },
    "unet_modern": {
        "xs": "32²",
        "s": "48²",
        "m": "64²",
        "l": "80²",
        "xl": "192²",
    },
}
# Probe H2D (StarPU GB) / nntile peak from the VRAM fit pass.
PROBE_BUS = {
    "lenet": {
        "xs": (6.5, 2.02),
        "s": (12.7, 4.01),
        "m": (19.6, 6.22),
        "l": (32.4, 10.41),
        "xl": (39.6, 12.64),
    },
    "resnet": {
        "xs": (5.2, 3.52),
        "s": (10.1, 6.41),
        "m": (19.9, 11.39),
        "l": (31.9, 17.02),
        "xl": (40.8, 8.90),
    },
    "vgg": {
        "xs": (6.7, 2.28),
        "s": (11.0, 3.50),
        "m": (21.2, 6.84),
        "l": (34.8, 11.28),
        "xl": (40.0, 7.00),
    },
    "mobilenet": {
        "xs": (5.2, 1.68),
        "s": (10.6, 3.49),
        "m": (18.3, 5.91),
        "l": (27.3, 8.51),
        "xl": (41.7, 7.80),
    },
    "unet": {
        "xs": (5.9, 1.85),
        "s": (13.0, 4.16),
        "m": (20.2, 6.50),
        "l": (29.0, 9.36),
        "xl": (43.5, 11.47),
    },
    "unet_modern": {
        "xs": (5.6, 1.73),
        "s": (12.2, 3.88),
        "m": (18.9, 6.06),
        "l": (27.1, 8.73),
        "xl": (41.9, 10.70),
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def grp(
    summary: dict[str, Any], family: str, size: str, device: str, mode: str
) -> dict[str, Any]:
    key = f"{family}_{size}_{device}_{mode}"
    groups = summary["groups"]
    if key not in groups:
        raise KeyError(f"missing group {key}")
    return groups[key]


def ms(stat: dict[str, float], ndigits: int = 3) -> str:
    mean = stat["mean"]
    std = stat["std"]
    if stat.get("n", 0) <= 1 or std == 0:
        return f"{mean:.{ndigits}f}"
    return f"{mean:.{ndigits}f} ± {std:.{ndigits}f}"


def ms_s(stat: dict[str, float], ndigits: int = 3) -> str:
    return f"{ms(stat, ndigits)} s"


def ratio_str(nntile: float, cuda: float) -> str:
    return f"**{nntile / cuda:.2f}×**"


def loss_str(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.6f}"


def vram_str(stat: dict[str, float] | None, fallback: float | None) -> str:
    if stat is not None:
        return f"{stat['mean']:.1f} GiB"
    if fallback is not None:
        return f"{fallback:.1f} GiB"
    return "TBD"


def metric(
    group: dict[str, Any], name: str
) -> dict[str, float] | None:
    return group.get("metrics", {}).get(name)


def iso(group: dict[str, Any], name: str) -> dict[str, float] | None:
    return group.get("isolated", {}).get(name)


def build_section(family: str, summary: dict[str, Any]) -> str:
    spatial = SPATIAL[family]
    bus = PROBE_BUS[family]
    lines: list[str] = [
        "## Two setups",
        "",
        "### Loss",
        "",
        "| Setup | HF(cuda) | HF(nntile) |",
        "|-------|-----:|----------------:|",
    ]
    for size in SIZES:
        cuda = grp(summary, family, size, "cuda", "overlap")
        nnt = grp(summary, family, size, "nntile", "overlap")
        label = f"{size.upper()} {spatial[size]}"
        lines.append(
            f"| {label} | {loss_str(metric(cuda, 'final_loss'))} | "
            f"{loss_str(metric(nnt, 'final_loss'))} |"
        )
    lines += [
        "",
        "### 10-step train wall",
        "",
        "**10 repeats** (mean ± stdev), `STARPU_LIMIT_CUDA_MEM=46000`.",
        "",
        "| Setup | HF(cuda) | HF(nntile) | HF(nntile) / HF(cuda) |",
        "|-------|-----:|---------:|--------------:|",
    ]
    for size in SIZES:
        cuda = grp(summary, family, size, "cuda", "overlap")
        nnt = grp(summary, family, size, "nntile", "overlap")
        cw = metric(cuda, "train_wall_s")
        nw = metric(nnt, "train_wall_s")
        label = f"{size.upper()} {spatial[size]}"
        lines.append(
            f"| {label} | {ms_s(cw)} | {ms_s(nw)} | "
            f"{ratio_str(nw['mean'], cw['mean'])} |"
        )
    lines += [
        "",
        "### Peak VRAM and bus",
        "",
        "10-step overlap, NVIDIA A40, `STARPU_LIMIT_CUDA_MEM=46000`, B=1. "
        "Peak VRAM is `nvidia-smi memory.used` polled by a sidecar process "
        "(`watch_gpu_peak.py`) while the train child runs "
        "(`peak_vram_gib=`). H2D/D2H are StarPU bus stats at shutdown "
        "from the VRAM-fit probe. **D2H is 0 on every size**.",
        "",
        "| Setup | HF(cuda) VRAM | HF(nntile) VRAM | H2D | D2H |",
        "|-------|----------:|----------------:|----:|----:|",
    ]
    for size in SIZES:
        cuda = grp(summary, family, size, "cuda", "overlap")
        nnt = grp(summary, family, size, "nntile", "overlap")
        probe_peak, h2d = bus[size]
        label = f"{size.upper()} {spatial[size]}"
        cuda_v = vram_str(metric(cuda, "peak_vram_gib"), None)
        nnt_v = vram_str(metric(nnt, "peak_vram_gib"), probe_peak)
        lines.append(
            f"| {label} | {cuda_v} | {nnt_v} | {h2d:.2f} GB | **0** |"
        )
    lines += [
        "",
        "## HF(nntile) vs HF(cuda) (10 repeats)",
        "",
        "Overlap mode. Host = `record(nntile)+record(torch)+compile`.",
        "",
        "| Setup | HF(cuda) wall | HF(nntile) wall | HF(nntile) / HF(cuda) | "
        "record(nntile) | record(torch) | compile | run | wait | "
        "host/wall | isolated |",
        "|-------|----------:|------------:|------------:|"
        "---------------:|--------------:|--------:|----:|-----:"
        "|----------:|---------:|",
    ]
    for size in SIZES:
        cuda = grp(summary, family, size, "cuda", "overlap")
        nnt = grp(summary, family, size, "nntile", "overlap")
        cw = metric(cuda, "train_wall_s")
        nw = metric(nnt, "train_wall_s")
        hf = metric(nnt, "host_frac")
        iso_rw = iso(nnt, "run_wait")
        label = f"{size.upper()} {spatial[size]}"
        iso_s = ms_s(iso_rw) if iso_rw else "TBD"
        host = f"**{hf['mean'] * 100:.1f}%**" if hf else "TBD"
        lines.append(
            f"| {label} | {ms_s(cw)} | {ms_s(nw)} | "
            f"{ratio_str(nw['mean'], cw['mean'])} | "
            f"{ms_s(metric(nnt, 'record_nntile_s'))} | "
            f"{ms_s(metric(nnt, 'record_torch_s'))} | "
            f"{ms_s(metric(nnt, 'compile_s'))} | "
            f"{ms_s(metric(nnt, 'run_s'))} | "
            f"{ms_s(metric(nnt, 'wait_s'))} | {host} | {iso_s} |"
        )
    lines += [
        "",
        "## Sequential HF(nntile)",
        "",
        "`--wait-after-run`: record → compile → run → wait (no overlap).",
        "",
        "| Setup | HF(cuda) | HF(nntile) overlap | HF(nntile) sequential | "
        "seq / cuda |",
        "|-------|-----:|---------:|----------:|----------:|",
    ]
    for size in SIZES:
        cuda = grp(summary, family, size, "cuda", "overlap")
        ov = grp(summary, family, size, "nntile", "overlap")
        seq = grp(summary, family, size, "nntile", "sequential")
        cw = metric(cuda, "train_wall_s")
        ow = metric(ov, "train_wall_s")
        sw = metric(seq, "train_wall_s")
        label = f"{size.upper()} {spatial[size]}"
        lines.append(
            f"| {label} | {ms_s(cw)} | {ms_s(ow)} | {ms_s(sw)} | "
            f"{ratio_str(sw['mean'], cw['mean'])} |"
        )
    long = None
    for mode in ("100step",):
        key = f"{family}_s_nntile_{mode}"
        if key in summary["groups"]:
            long = summary["groups"][key]
            break
    if long is not None:
        lw = metric(long, "train_wall_s")
        ll = metric(long, "final_loss")
        hf = metric(long, "host_frac")
        host = f"{hf['mean'] * 100:.1f}%" if hf else "n/a"
        lines += [
            "",
            "## S HF(nntile) 100-step",
            "",
            f"Overlap, size S, 100 steps, 10 repeats: wall "
            f"{ms_s(lw)}, loss {loss_str(ll)}, host/wall {host}.",
        ]
    lines.append("")
    return "\n".join(lines)


def replace_two_setups(doc: Path, section: str) -> None:
    text = doc.read_text(encoding="utf-8")
    start = text.find("## Two setups")
    end = text.find("## How to reproduce")
    if start < 0 or end < 0 or end <= start:
        raise SystemExit(f"could not find Two setups / How to reproduce in {doc}")
    new = text[:start] + section + "\n" + text[end:]
    new = new.replace(
        "**Placeholder.** Wall / loss tables are still empty — fill those after\n"
        "the 10-repeat ladder in [How to reproduce](#how-to-reproduce).\n"
        "**VRAM is confirmed** (see [Peak VRAM](#peak-vram-and-bus)).\n\n",
        "",
    )
    doc.write_text(new, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=list(SPATIAL))
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--doc",
        type=Path,
        default=None,
        help="Markdown to patch (default: docs/dev/<family>_hf_overhead_scale.md)",
    )
    args = parser.parse_args()
    doc = args.doc or (
        REPO / "docs" / "dev" / f"{args.family}_hf_overhead_scale.md"
    )
    summary = load_json(args.summary)
    section = build_section(args.family, summary)
    replace_two_setups(doc, section)
    print(f"updated {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
