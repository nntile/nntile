#!/usr/bin/env python3
"""Fill docs/dev/<family>_hf_overhead_scale.md from a CNN ladder summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from overhead_plot import write_long_plots

REPO = Path(__file__).resolve().parents[2]
FAMILY_TITLE = {
    "lenet": "LeNet",
    "resnet": "ResNet",
    "vgg": "VGG",
    "mobilenet": "MobileNet",
    "unet": "U-Net",
    "unet_modern": "Modern U-Net",
}
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
    long = long_group(family, summary)
    if long is not None:
        lines += ["", build_long_section(family, summary, long)]
    lines.append("")
    return "\n".join(lines)


def long_group(
    family: str, summary: dict[str, Any]
) -> dict[str, Any] | None:
    for mode in ("100step",):
        key = f"{family}_s_nntile_{mode}"
        if key in summary.get("groups", {}):
            return summary["groups"][key]
    return None


def build_long_section(
    family: str,
    summary: dict[str, Any],
    long: dict[str, Any],
) -> str:
    long_steps = int(summary.get("long_steps", 100))
    repeats = int(summary.get("repeats", long.get("n", 0)))
    lw = metric(long, "train_wall_s")
    ll = metric(long, "final_loss")
    hf = metric(long, "host_frac")
    host = f"{hf['mean'] * 100:.0f}%" if hf else "n/a"
    lines = [
        f"## S HF(nntile) {long_steps}-step",
        "",
        f"Overlap, size S, {long_steps} steps, {repeats} repeats "
        f"(mean ± stdev). Complements the 10-step HF ladder above.",
        "",
        f"Loss **{loss_str(ll)}**.",
        "",
        "| | Total | mean / step |",
        "|--|--:|--:|",
    ]
    for name, key in (
        ("record(nntile)", "record_nntile_s"),
        ("record(torch)", "record_torch_s"),
        ("compile", "compile_s"),
        ("run", "run_s"),
        ("wait", "wait_s"),
    ):
        stat = metric(long, key)
        if stat is None:
            continue
        per = stat["mean"] / long_steps * 1000
        per_s = f"{per:.1f} ms" if per < 10 else f"{per:.0f} ms"
        lines.append(f"| {name} | {ms_s(stat)} | {per_s} |")
    if lw is not None:
        per = lw["mean"] / long_steps * 1000
        per_s = f"{per:.1f} ms" if per < 10 else f"{per:.0f} ms"
        lines.append(f"| **train wall** | **{ms_s(lw)}** | {per_s} |")
    lines += [
        "",
        f"Host (record + compile) is **{host}** of the wall.",
        "",
        f"![Host overhead per iteration]"
        f"({family}_hf_overhead_s_{long_steps}.svg)",
        "",
        f"CSV: [`{family}_hf_overhead_s_{long_steps}.csv`]"
        f"({family}_hf_overhead_s_{long_steps}.csv) "
        f"(median of {repeats} runs).",
    ]
    return "\n".join(lines)


def replace_long_section(doc: Path, section: str) -> None:
    text = doc.read_text(encoding="utf-8")
    start = text.find("## S HF(nntile) 100-step")
    if start < 0:
        raise SystemExit(f"could not find 100-step section in {doc}")
    end = text.find("## How to reproduce", start)
    if end < 0 or end <= start:
        raise SystemExit(
            f"could not find How to reproduce after 100-step in {doc}"
        )
    doc.write_text(
        text[:start] + section + "\n\n" + text[end:], encoding="utf-8"
    )


def write_family_long_plots(
    family: str,
    results: list[dict[str, Any]],
    long_steps: int,
) -> bool:
    title = FAMILY_TITLE[family]
    csv_path = REPO / "docs" / "dev" / f"{family}_hf_overhead_s_{long_steps}.csv"
    svg_path = REPO / "docs" / "dev" / f"{family}_hf_overhead_s_{long_steps}.svg"
    ok = write_long_plots(
        results,
        long_mode=f"{long_steps}step",
        csv_path=csv_path,
        svg_path=svg_path,
        title=(
            f"{title} S HF(nntile) host overhead per iteration "
            f"({long_steps} steps)"
        ),
    )
    if ok:
        print(f"wrote {csv_path}")
        print(f"wrote {svg_path}")
    return ok


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
        "--results",
        type=Path,
        default=None,
        help="results.json with per-iter 100-step rows (default: beside summary)",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=None,
        help="Markdown to patch (default: docs/dev/<family>_hf_overhead_scale.md)",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Refresh the 100-step section and SVG/CSV; leave 10-step tables",
    )
    args = parser.parse_args()
    doc = args.doc or (
        REPO / "docs" / "dev" / f"{args.family}_hf_overhead_scale.md"
    )
    summary = load_json(args.summary)
    long_steps = int(summary.get("long_steps", 100))
    results_path = args.results or (args.summary.parent / "results.json")
    if args.plots_only:
        long = long_group(args.family, summary)
        if long is None:
            raise SystemExit(
                f"no {args.family} S nntile {long_steps}step group in "
                f"{args.summary}"
            )
        replace_long_section(
            doc, build_long_section(args.family, summary, long)
        )
        print(f"updated {doc} (100-step section)")
    else:
        section = build_section(args.family, summary)
        replace_two_setups(doc, section)
        print(f"updated {doc}")
    if results_path.is_file():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        write_family_long_plots(args.family, results, long_steps)
    else:
        print(f"skip plots (missing {results_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
