#!/usr/bin/env python3
"""CSV + SVG plots for GPT-2 / GPT-Neo overhead 100-step runs."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any


def write_long_csv(
    results: list[dict[str, Any]],
    long_mode: str,
    csv_path: Path,
) -> bool:
    long = [r for r in results if r.get("mode") == long_mode]
    if not long:
        return False
    med = statistics.median([r["train_wall_s"] for r in long])
    best = min(long, key=lambda r: abs(r["train_wall_s"] - med))
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "step,record_nntile_s,record_torch_s,compile_s,run_s,wait_s\n"
        )
        for it in best["iters"]:
            f.write(
                ",".join(
                    [
                        str(it["step"]),
                        f"{it['record_nntile']:.3f}",
                        f"{it['record_torch']:.3f}",
                        f"{it['compile']:.3f}",
                        f"{it['run']:.3f}",
                        f"{it['wait']:.3f}",
                    ]
                )
                + "\n"
            )
    return True


def write_host_overhead_svg(
    csv_path: Path,
    svg_path: Path,
    *,
    title: str,
    y_clip_ms: float = 60.0,
) -> None:
    rows: list[dict[str, float]] = []
    for line in csv_path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        step, rn, rt, comp, run, _wait = line.split(",")
        rows.append(
            {
                "step": int(step),
                "record_nntile": float(rn) * 1000,
                "record_torch": float(rt) * 1000,
                "compile": float(comp) * 1000,
                "run": float(run) * 1000,
            }
        )
    if not rows:
        return

    width, height = 920, 420
    left, top, right, bottom = 62, 36, 902, 372
    plot_w = right - left
    plot_h = bottom - top
    n = len(rows)
    ymax = y_clip_ms
    spikes: list[str] = []

    def xy(i: int, val_ms: float) -> tuple[float, float]:
        x = left + (i / max(n - 1, 1)) * plot_w
        y = bottom - min(val_ms, ymax) / ymax * plot_h
        return x, y

    series = [
        ("record(torch)", "record_torch", "#d95f02"),
        ("record(nntile)", "record_nntile", "#1b9e77"),
        ("compile", "compile", "#7570b3"),
        ("run", "run", "#e7298a"),
    ]
    polylines: list[str] = []
    for _name, key, color in series:
        pts = []
        for i, row in enumerate(rows):
            val = row[key]
            if val > ymax:
                spikes.append(f"iter {row['step']} {key} {val:.0f} ms")
            x, y = xy(i, val)
            pts.append(f"{x:.1f},{y:.1f}")
        polylines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="1.4" '
            f'stroke-linejoin="round" points="{" ".join(pts)}"/>'
        )

    y_ticks = []
    for tick in range(0, int(ymax) + 1, 10):
        y = bottom - tick / ymax * plot_h
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            f'stroke="#e6e6e6" stroke-width="1"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="12" font-family="sans-serif" fill="#444">{tick}</text>'
        )

    x_labels = {1}
    for mark in (10, 25, 50, 75, 100):
        if mark <= n:
            x_labels.add(mark)
    x_labels.add(n)
    x_ticks = []
    for step in sorted(x_labels):
        i = step - 1
        x = left + (i / max(n - 1, 1)) * plot_w
        x_ticks.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" '
            f'stroke="#f0f0f0" stroke-width="1"/>'
            f'<text x="{x:.1f}" y="{bottom + 18:.1f}" text-anchor="middle" '
            f'font-size="12" font-family="sans-serif" fill="#444">{step}</text>'
        )

    legend = []
    ly = 50
    for name, _key, color in series:
        legend.append(
            f'<line x1="70" y1="{ly}" x2="92" y2="{ly}" stroke="{color}" '
            f'stroke-width="2.4"/>'
            f'<text x="98" y="{ly + 4}" font-size="13" font-family="sans-serif" '
            f'fill="#222">{name}</text>'
        )
        ly += 16

    spike_note = ""
    if spikes:
        uniq = ", ".join(spikes[:3])
        if len(spikes) > 3:
            uniq += f", … ({len(spikes)} clipped)"
        spike_note = (
            f'<text x="{width / 2:.1f}" y="412" text-anchor="middle" '
            f'font-size="11" font-family="sans-serif" fill="#666">'
            f"y clipped at {ymax:.0f} ms; spikes: {uniq}</text>"
        )

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.1f}" y="22" text-anchor="middle" font-size="15" font-family="sans-serif" font-weight="600" fill="#111">{title}</text>
  {''.join(y_ticks)}
  {''.join(x_ticks)}
  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333" stroke-width="1"/>
  {''.join(polylines)}
  {''.join(legend)}
  <text x="{width / 2:.1f}" y="396" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#333">iteration</text>
  <text x="16" y="{height / 2:.1f}" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#333" transform="rotate(-90 16 {height / 2:.1f})">time (ms)</text>
  {spike_note}
</svg>
"""
    svg_path.write_text(svg, encoding="utf-8")


def write_long_plots(
    results: list[dict[str, Any]],
    *,
    long_mode: str,
    csv_path: Path,
    svg_path: Path,
    title: str,
) -> bool:
    if not write_long_csv(results, long_mode, csv_path):
        return False
    write_host_overhead_svg(csv_path, svg_path, title=title)
    return True
