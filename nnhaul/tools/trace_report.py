#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# NNHaul is a task-based dynamic execution engine over N memory nodes:
# a virtual planner builds one plan per batch from a drained snapshot,
# and a literal executor issues exactly the planned operations on the
# lanes of the node inventory.
#
# @file tools/trace_report.py
# Offline trace viewer: reads one NNHaul Chrome Trace Event JSON
# (format_version 1, the chunk-graph tracer's dump) and writes a single
# self-contained HTML report. One swimlane per LANE of the trace's
# inventory (exec(node) and chan(a,b) alike, from the metadata lane
# frame) is drawn as inline SVG over a shared time axis, with the plan
# witness (batches, planned == issued, reconcile) in the header. The
# previous format's fixed three streams and vram/ram counter charts
# died with the reactive engine. No network, no CDN, no scripts, no
# external assets: pure inline SVG that renders straight from file://.
# Output is deterministic - the same input yields byte-identical HTML
# (ids/values are derived from content only; no timestamps, no
# randomness).
#
# Usage:
#   python3 tools/trace_report.py <trace.json> [-o trace.html]
#
# @version 1.0.0

import argparse
import html
import json
import math
import sys

PROG = "trace_report.py"
FORMAT_VERSION = 1
COPY_COLOR = "#f58518"
TILE_OP_COLOR = "#4c78a8"
OTHER_COLOR = "#525252"

# SVG layout constants (pixels).
SVG_W = 980
PLOT_X0 = 150
PLOT_X1 = SVG_W - 16
PLOT_W = PLOT_X1 - PLOT_X0
AXIS_H = 26
LANE_ROW_H = 14
LANE_PAD = 6
LANE_GAP = 12
LANE_MAX_ROWS = 4

CSS = """body{font-family:system-ui,sans-serif;color:#222;background:#fff;
margin:24px auto;max-width:1020px;padding:0 12px;}
h1{font-size:20px;margin:0 0 12px 0;}
dl.meta{display:grid;grid-template-columns:max-content max-content;
gap:2px 20px;font-size:13px;margin:0 0 6px 0;}
dl.meta dt{font-weight:600;color:#555;}
dl.meta dd{margin:0;}
p.counts{font-size:13px;color:#555;margin:0 0 16px 0;}
p.note{font-size:11px;color:#777;margin:10px 0 0 0;max-width:960px;}
svg{display:block;}
text{font-family:system-ui,sans-serif;}
.lanelabel{font-size:12px;fill:#333;text-anchor:end;}
.lanesub{font-size:10px;fill:#888;text-anchor:end;}
.lanebg{fill:#f4f4f4;stroke:#ddd;}
.iv-copy{fill:#f58518;fill-opacity:.85;stroke:#b0600f;}
.iv-tile{fill:#4c78a8;fill-opacity:.85;stroke:#33587c;}
.iv-other{fill:#525252;fill-opacity:.85;stroke:#333;}
.axis line{stroke:#999;stroke-width:1;}
.gridline{stroke:#e4e4e4;stroke-width:1;}
.ticklabel{font-size:10px;fill:#555;text-anchor:middle;}
.axiscap{font-size:10px;fill:#777;text-anchor:end;}
.charttitle{font-size:12px;fill:#333;font-weight:600;}
.empty{font-size:11px;fill:#999;font-style:italic;text-anchor:middle;}
"""


def die(msg):
    print("%s: error: %s" % (PROG, msg), file=sys.stderr)
    sys.exit(1)


def esc(s):
    return html.escape(str(s), quote=True)


def num(v, default=0.0):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return default
    return float(v)


def fmt_us(us):
    us = max(float(us), 0.0)
    if us >= 1000000.0:
        return "%.3f s" % (us / 1000000.0)
    if us >= 1000.0:
        return "%.3f ms" % (us / 1000.0)
    return "%d µs" % int(round(us))


def tick_unit(t_max):
    """Pick µs / ms / s for the whole axis from its extent."""
    if t_max >= 1000000.0:
        return "s"
    if t_max >= 1000.0:
        return "ms"
    return "µs"


def fmt_tick(v, unit):
    v = max(float(v), 0.0)
    if unit == "s":
        s = "%.2f" % (v / 1000000.0)
    elif unit == "ms":
        s = "%.2f" % (v / 1000.0)
    else:
        return "%d µs" % int(round(v))
    s = s.rstrip("0").rstrip(".") or "0"
    return "%s %s" % (s, unit)


def nice_ticks(t_max, target=8):
    """Deterministic 'nice' tick positions covering [0, t_max]."""
    if t_max <= 0.0:
        return [0.0], 0.0
    raw = t_max / max(target, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    step = 10.0 * mag
    for mult in (1.0, 2.0, 5.0, 10.0):
        if mag * mult >= raw:
            step = mag * mult
            break
    count = int(math.floor(t_max / step + 1e-9))
    return [i * step for i in range(count + 1)], step


def load_trace(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except OSError as exc:
        die("cannot read %s: %s" % (path, exc.strerror or exc))
    except ValueError as exc:
        die("%s is not valid JSON: %s" % (path, exc))
    if not isinstance(doc, dict):
        die("%s: top-level JSON value is not an object" % path)
    events = doc.get("traceEvents")
    if not isinstance(events, list):
        die("%s: missing (or non-array) 'traceEvents'" % path)
    meta = doc.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    # Unknown extra top-level keys (e.g. "_comment") are ignored.
    return events, meta


def collect(events, meta):
    """Split events into per-lane interval lists over the lane frame."""
    thread_names = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("ph") == "M" and ev.get("name") == "thread_name":
            args = ev.get("args")
            name = args.get("name") if isinstance(args, dict) else None
            tid = ev.get("tid")
            if isinstance(name, str) and tid is not None:
                thread_names[tid] = name
    lanes = []
    meta_lanes = meta.get("lanes")
    if isinstance(meta_lanes, list) and meta_lanes:
        for entry in meta_lanes:
            if isinstance(entry, dict) and "tid" in entry:
                tid = entry.get("tid")
                lanes.append({
                    "tid": tid,
                    "lane": entry.get("lane"),
                    "name": str(entry.get("name",
                                thread_names.get(tid, "tid %s" % tid))),
                })
    else:
        seen = dict(thread_names)
        for ev in events:
            if isinstance(ev, dict) and ev.get("ph") == "X" and \
                    "tid" in ev:
                seen.setdefault(ev["tid"], "tid %s" % ev["tid"])
        lanes = [{"tid": tid, "lane": None, "name": name}
                 for tid, name in sorted(seen.items())]
    by_tid = {lane["tid"]: [] for lane in lanes}
    ph_counts = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        ph = ev.get("ph")
        ph_counts[ph] = ph_counts.get(ph, 0) + 1
        if ph == "X" and ev.get("tid") in by_tid:
            args = ev.get("args")
            kind = args.get("kind") if isinstance(args, dict) else None
            by_tid[ev["tid"]].append(
                (num(ev.get("ts")), num(ev.get("dur")), kind, ev))
    for intervals in by_tid.values():  # stable for equal ts
        intervals.sort(key=lambda item: item[0])
    for lane in lanes:
        lane["intervals"] = by_tid[lane["tid"]]
    return lanes, ph_counts


def compute_span(meta, lanes):
    """Axis span: metadata wall_clock_span_us, else derived max ts.

    Returns (span, t_max, derived): t_max additionally covers interval
    ends (ts + dur) so no rectangle is ever clipped off the right edge.
    """
    raw = meta.get("wall_clock_span_us")
    derived = not (isinstance(raw, (int, float)) and not isinstance(raw, bool))
    span = num(raw) if not derived else 0.0
    if derived:
        for lane in lanes:
            for ts, _dur, _kind, _ev in lane["intervals"]:
                span = max(span, ts)
    t_max = span
    for lane in lanes:
        for ts, dur, _kind, _ev in lane["intervals"]:
            t_max = max(t_max, ts + max(dur, 0.0))
    return span, t_max, derived


def lane_rows(intervals):
    """Greedy first-fit row packing; same-lane overlaps get their own
    row. Row count is capped at LANE_MAX_ROWS."""
    row_ends = []
    placement = []
    for ts, dur, _kind, _ev in intervals:
        end = ts + max(dur, 0.0)
        row = len(row_ends)
        for i, last in enumerate(row_ends):
            if ts >= last:
                row_ends[i] = end
                row = i
                break
        if row == len(row_ends):
            row_ends.append(end)
        placement.append(min(row, LANE_MAX_ROWS - 1))
    n_rows = min(max(len(row_ends), 1), LANE_MAX_ROWS)
    return placement, n_rows


def render_axis(y, ticks, unit, x_of):
    parts = ['<g class="axis">']
    parts.append('<line x1="%d" y1="%.2f" x2="%d" y2="%.2f"/>' % (
        PLOT_X0, y, PLOT_X1, y))
    parts.append('<text class="axiscap" x="%d" y="%.2f">time</text>' % (
        PLOT_X1, y - 5.0))
    for t in ticks:
        x = min(x_of(t), float(PLOT_X1))
        parts.append('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>' % (
            x, y, x, y + 5.0))
        parts.append('<text class="ticklabel" x="%.2f" y="%.2f">%s</text>' % (
            x, y + 17.0, esc(fmt_tick(t, unit))))
    parts.append('</g>')
    return parts


def interval_title(ev, kind, dur):
    lines = [str(ev.get("name", "interval"))]
    lines.append("dur %s" % fmt_us(dur))
    if kind is not None:
        lines.append("kind %s" % kind)
    args = ev.get("args")
    args = args if isinstance(args, dict) else {}
    if isinstance(args.get("plan_seq"), (int, float)) and not \
            isinstance(args.get("plan_seq"), bool):
        lines.append("plan %s position %s" % (
            int(args["plan_seq"]), int(args.get("position", 0))))
    if isinstance(args.get("lane_name"), str):
        lines.append("lane %s" % args["lane_name"])
    return "\n".join(lines)


def kind_class(kind):
    if kind == "copy":
        return "iv-copy", COPY_COLOR
    if kind == "tile_op":
        return "iv-tile", TILE_OP_COLOR
    return "iv-other", OTHER_COLOR


def render_lane(y, lane, x_of, t_draw, ticks):
    intervals = lane["intervals"]
    if intervals:
        placement, n_rows = lane_rows(intervals)
        lane_h = n_rows * LANE_ROW_H + 2 * LANE_PAD
    else:
        placement, n_rows, lane_h = [], 1, LANE_ROW_H + 2 * LANE_PAD
    lane_id = "lane-%s" % (lane["lane"] if lane["lane"] is not None
                           else "tid%s" % lane["tid"])
    parts = ['<g class="lane" id="%s">' % esc(lane_id)]
    parts.append('<rect class="lanebg" x="%d" y="%.2f" width="%d" '
                 'height="%.2f"/>' % (PLOT_X0, y, PLOT_W, lane_h))
    for t in ticks:
        x = min(x_of(t), float(PLOT_X1))
        parts.append('<line class="gridline" x1="%.2f" y1="%.2f" x2="%.2f" '
                     'y2="%.2f"/>' % (x, y, x, y + lane_h))
    cy = y + lane_h / 2.0
    parts.append('<text class="lanelabel" x="%d" y="%.2f">%s</text>' % (
        PLOT_X0 - 10, cy - 1.0, esc(lane["name"])))
    sub = "%d event%s" % (len(intervals),
                          "" if len(intervals) == 1 else "s")
    if intervals and t_draw > 0.0:
        busy = sum(d for _ts, d, _kind, _ev in intervals) / t_draw * 100.0
        sub += " · %d%% busy" % int(round(busy))
    parts.append('<text class="lanesub" x="%d" y="%.2f">%s</text>' % (
        PLOT_X0 - 10, cy + 10.0, esc(sub)))
    if not intervals:
        parts.append('<text class="empty" x="%.2f" y="%.2f">no events'
                     '</text>' % ((PLOT_X0 + PLOT_X1) / 2.0, cy + 4.0))
        parts.append('</g>')
        return parts, lane_h
    for (ts, dur, kind, ev), row in zip(intervals, placement):
        x = max(x_of(ts), float(PLOT_X0))
        xr = min(x_of(ts + max(dur, 0.0)), float(PLOT_X1))
        w = max(xr - x, 1.0)
        ry = y + LANE_PAD + row * LANE_ROW_H + 1.0
        cls, _color = kind_class(kind)
        parts.append('<rect class="%s" x="%.2f" y="%.2f" width="%.2f" '
                     'height="%.2f"><title>%s</title></rect>' % (
                         cls, x, ry, w, LANE_ROW_H - 2.0,
                         esc(interval_title(ev, kind, dur))))
    parts.append('</g>')
    return parts, lane_h


def meta_str(meta, key, default):
    val = meta.get(key)
    return str(val) if val is not None else default


def build_html(events, meta, lanes, span, t_max, derived, ph_counts):
    t_draw = t_max if t_max > 0.0 else 1.0

    def x_of(us):
        return PLOT_X0 + (max(us, 0.0) / t_draw) * PLOT_W

    ticks, _step = nice_ticks(t_max)
    unit = tick_unit(t_max)

    # ---- HTML head + metadata header -------------------------------
    out = []
    out.append("<!DOCTYPE html>")
    out.append("<!-- SPDX-License-Identifier: MIT -->")
    out.append('<html lang="en">')
    out.append("<head>")
    out.append('<meta charset="utf-8">')
    out.append("<title>NNHaul trace report</title>")
    out.append("<!-- Generated by tools/trace_report.py; self-contained "
               "offline SVG report. @version 1.0.0 -->")
    out.append("<style>")
    out.append(CSS)
    out.append("</style>")
    out.append("</head>")
    out.append("<body>")
    out.append("<h1>NNHaul trace report</h1>")
    span_text = fmt_us(span)
    if float(span) == int(span):
        span_text += " (%d µs)" % int(span)
    if derived:
        span_text += ", derived from event timestamps"
    plan = meta.get("plan")
    plan = plan if isinstance(plan, dict) else {}
    if plan.get("reconcile_ok") is True:
        witness = "planned %s == issued %s" % (
            plan.get("planned_ops"), plan.get("issued_ops"))
    elif plan:
        witness = "planned %s, issued %s (reconcile FAILED)" % (
            plan.get("planned_ops"), plan.get("issued_ops"))
    else:
        witness = "not reported"
    rows = [
        ("runtime version", meta_str(meta, "runtime_version", "unknown")),
        ("format version", meta_str(meta, "format_version", "unknown")),
        ("lanes", "%d" % len(lanes)),
        ("span", span_text),
        ("plan witness", witness),
    ]
    out.append('<dl class="meta">')
    for dt, dd in rows:
        out.append("<dt>%s</dt><dd>%s</dd>" % (esc(dt), esc(dd)))
    out.append("</dl>")
    x_by_lane = " · ".join("%s %d" % (lane["name"], len(lane["intervals"]))
                           for lane in lanes)
    counts = "%d trace events: %d M · %d X (%s)" % (
        len(events), ph_counts.get("M", 0), ph_counts.get("X", 0),
        x_by_lane if x_by_lane else "none")
    out.append('<p class="counts">%s</p>' % esc(counts))

    # ---- SVG: shared axis, one swimlane per lane of the frame, axis --
    frag = []
    y = 16.0
    frag += render_axis(y, ticks, unit, x_of)
    y += AXIS_H + 8.0
    for lane in lanes:
        parts, lane_h = render_lane(y, lane, x_of, t_draw, ticks)
        frag += parts
        y += lane_h + LANE_GAP
    y -= LANE_GAP + 4.0
    frag += render_axis(y, ticks, unit, x_of)
    y += AXIS_H + 6.0
    height = int(math.ceil(y))
    out.append('<svg width="%d" height="%d" viewBox="0 0 %d %d" role="img" '
               'aria-label="NNHaul trace swimlane timeline over the '
               'inventory lanes">' % (SVG_W, height, SVG_W, height))
    out += frag
    out.append("</svg>")
    out.append('<p class="note">Timestamps are steady-clock microseconds '
               "read at each op's issue position; an interval runs from "
               "that stamp to the next stamp on the lane, so it includes "
               "queue wait, not only device work. Each row is one lane of "
               "the run's inventory (exec(node) and chan(a,b) alike), "
               "an empty lane a finding in itself. Orange intervals are "
               "planned copy ops, blue ones tile ops. Open offline, "
               "straight from the filesystem.</p>")
    out.append("</body>")
    out.append("</html>")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog=PROG,
        description="Render a NNHaul Chrome Trace Event JSON (format "
                    "version 2) as a single self-contained HTML swimlane "
                    "report (inline SVG, no network, renders from "
                    "file://).")
    ap.add_argument("trace", help="input trace JSON (Chrome Trace Event "
                    "format, as dumped by the NNHaul tracer)")
    ap.add_argument("-o", "--output", default="trace.html",
                    help="output HTML path (default: %(default)s)")
    args = ap.parse_args(argv)

    events, meta = load_trace(args.trace)
    lanes, ph_counts = collect(events, meta)
    span, t_max, derived = compute_span(meta, lanes)
    doc = build_html(events, meta, lanes, span, t_max, derived, ph_counts)
    try:
        with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(doc)
    except OSError as exc:
        die("cannot write %s: %s" % (args.output, exc.strerror or exc))
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
