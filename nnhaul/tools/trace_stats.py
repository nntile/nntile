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
# @file tools/trace_stats.py
# Chrome-Trace analyzer for NNHaul trace dumps (format_version 1): the
# wall span, per-lane busy fractions and counts over the
# inventory-derived lane set, the per-lane copy byte sums (the dumps'
# copy events carry each copy's real transferred bytes — the occupied
# span of its chunk — in args.bytes), the
# exec lanes' internal idle bubbles, op counts by kind, and the plan
# witness echo. Python 3, standard library only.
#
# @version 2.0.0

"""Analyze an NNHaul Chrome Trace Event dump (format_version 1).

Reads the JSON written by the NNHaul tracer and reports scheduling
statistics over the LANE SET of the trace's inventory (one lane per
metadata.lanes entry: exec(node) and chan(a,b) lanes alike, both named
by the inventory, never a vendor) and echoes the plan witness. Unknown
extra keys and events are tolerated.
"""

import argparse
import json
import sys

KINDS = ("copy", "tile_op")


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def fail(message):
    print(f"trace_stats: error: {message}", file=sys.stderr)
    sys.exit(1)


def load_trace(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    except json.JSONDecodeError as exc:
        fail(f"{path} is not valid JSON: {exc}")
    if not isinstance(data, dict) or not isinstance(
            data.get("traceEvents"), list):
        fail(f"{path}: missing 'traceEvents' array")
    return data


def lane_intervals(events, tid):
    """Sorted (start, end, event) intervals of one lane; dur 0 allowed."""
    intervals = []
    for event in events:
        if event.get("ph") != "X" or event.get("tid") != tid:
            continue
        ts, dur = event.get("ts"), event.get("dur")
        if not is_number(ts) or not is_number(dur):
            continue
        start = float(ts)
        # dur is fractional us (ns resolution in current dumps, whole
        # us in fixtures); float keeps both without re-quantising.
        intervals.append((start, start + float(dur), event))
    intervals.sort(key=lambda item: (item[0], item[1]))
    return intervals


def merge_intervals(intervals):
    """Union of the interval set as a list of disjoint (start, end)."""
    merged = []
    for start, end, _ in intervals:
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def busy_us(merged):
    return sum(end - start for start, end in merged)


def wall_span(meta):
    value = meta.get("wall_clock_span_us")
    if is_number(value) and value > 0:
        return float(value)
    return 0.0


def lane_table(meta, events):
    """The lane rows: metadata.lanes order, thread_name fallback."""
    meta_lanes = meta.get("lanes")
    thread_names = {}
    for event in events:
        if isinstance(event, dict) and event.get("ph") == "M" and \
                event.get("name") == "thread_name":
            args = event.get("args")
            name = args.get("name") if isinstance(args, dict) else None
            tid = event.get("tid")
            if isinstance(name, str) and is_number(tid):
                thread_names[int(tid)] = name
    rows = []
    if isinstance(meta_lanes, list):
        for entry in meta_lanes:
            if not isinstance(entry, dict):
                continue
            tid, lane = entry.get("tid"), entry.get("lane")
            if not is_number(tid) or not is_number(lane):
                continue
            rows.append({"tid": int(tid), "lane": int(lane),
                         "name": str(entry.get("name",
                                    thread_names.get(int(tid), "?")))})
    if not rows:  # no metadata frame: derive lanes from the events.
        seen = {}
        for tid, name in thread_names.items():
            seen[tid] = name
        for event in events:
            if isinstance(event, dict) and event.get("ph") == "X" and \
                    is_number(event.get("tid")):
                seen.setdefault(int(event["tid"]),
                                f"tid {int(event['tid'])}")
        rows = [{"tid": tid, "lane": tid - 1, "name": name}
                for tid, name in sorted(seen.items())]
    return rows


def find_bubbles(merged):
    """Maximal gaps strictly between consecutive intervals of one lane.

    A trailing gap after the last interval is NOT a bubble: no work is
    outstanding after the last op.
    """
    bubbles = []
    for (_, gap_start), (gap_end, _) in zip(merged, merged[1:]):
        if gap_end > gap_start:
            bubbles.append((gap_start, gap_end))
    return bubbles


def compute_stats(data):
    events = data["traceEvents"]
    meta = data.get("metadata") or {}
    span = wall_span(meta)
    rows = lane_table(meta, events)
    kind_counts = {}
    x_total = 0
    for row in rows:
        intervals = lane_intervals(events, row["tid"])
        merged = merge_intervals(intervals)
        row["events"] = len(intervals)
        row["busy_us"] = busy_us(merged)
        row["utilization"] = (row["busy_us"] / span
                              if span > 0 and row["busy_us"] else 0.0)
        row["is_exec"] = row["name"].startswith("exec(")
        row["bubbles"] = find_bubbles(merged) if row["is_exec"] else []
        # The byte facts: the sum of the lane's events' args.bytes —
        # the copy quantum the dumper stamps on every copy event, so a
        # transfer lane's sum IS its direction's copied bytes; an
        # event without the arg contributes nothing.
        row["bytes"] = 0
        for _start, _end, event in intervals:
            kind = (event.get("args") or {}).get("kind")
            if kind in KINDS:
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
            nbytes = (event.get("args") or {}).get("bytes")
            if is_number(nbytes) and not isinstance(nbytes, bool):
                row["bytes"] += nbytes
        x_total += len(intervals)
    lanes_out = [{
        "lane": row["lane"],
        "tid": row["tid"],
        "name": row["name"],
        "events": row["events"],
        "busy_us": row["busy_us"],
        "utilization": row["utilization"],
        "bytes": row["bytes"],
    } for row in rows]
    bubbles_out = []
    for row in rows:
        for start, end in row["bubbles"]:
            bubbles_out.append({
                "lane": row["lane"],
                "lane_name": row["name"],
                "start_us": start,
                "end_us": end,
                "dur_us": end - start,
            })
    bubbles_out.sort(key=lambda bubble: (-bubble["dur_us"],
                                         bubble["start_us"]))
    plan = meta.get("plan") if isinstance(meta.get("plan"), dict) else {}
    return {
        "span_us": int(span) if span == int(span) else span,
        "lane_count": len(rows),
        "x_events": x_total,
        "lanes": lanes_out,
        "bubbles": bubbles_out[:10],
        "ops": {kind: kind_counts.get(kind, 0) for kind in KINDS},
        "plan": {
            "batches": plan.get("batches"),
            "planned_ops": plan.get("planned_ops"),
            "issued_ops": plan.get("issued_ops"),
            "fired_ops": plan.get("fired_ops"),
            "reconcile_ok": plan.get("reconcile_ok"),
        },
    }


def print_human(path, stats):
    span = stats["span_us"]
    print(f"NNHaul trace: {path}")
    print(f"Wall span: {span} us ({span / 1000.0:.3f} ms)")
    print()
    print(f"Lane utilization ({stats['lane_count']} lanes; busy fraction"
          " of the wall span):")
    for lane in stats["lanes"]:
        line = (f"  {lane['name']:<10} {lane['utilization'] * 100:6.2f} %"
                f"  ({lane['events']} event(s),"
                f" {lane['busy_us']:.3f} us busy")
        if lane["bytes"]:
            line += f", {lane['bytes'] / 1048576.0:.1f} MiB copied"
        line += ")"
        print(line)
    print()
    bubbles = stats["bubbles"]
    print(f"Exec-lane idle bubbles (top {len(bubbles)}; internal gaps"
          " with ops still outstanding):")
    if bubbles:
        print("  #   lane        start_us       end_us         dur_us")
        for index, bubble in enumerate(bubbles, 1):
            print(f"  {index:<4}{bubble['lane_name']:<12}"
                  f"{bubble['start_us']:<15.3f}{bubble['end_us']:<15.3f}"
                  f"{bubble['dur_us']:.3f}")
    else:
        print("  (none)")
    print()
    print("Ops by kind (the X events' args.kind):")
    for kind, count in stats["ops"].items():
        print(f"  {kind:<8} {count}")
    print()
    print("Copy bytes per transfer lane (the copy events' args.bytes):")
    any_bytes = False
    for lane in stats["lanes"]:
        if lane["bytes"]:
            any_bytes = True
            print(f"  {lane['name']:<10} {lane['bytes']} B"
                  f" ({lane['bytes'] / 1048576.0:.1f} MiB)")
    if not any_bytes:
        print("  (none carried)")
    plan = stats["plan"]
    print()
    print("Plan witness (metadata.plan):")
    print(f"  batches {plan['batches']}, planned {plan['planned_ops']},"
          f" issued {plan['issued_ops']}, fired {plan['fired_ops']},"
          f" reconcile_ok {plan['reconcile_ok']}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="trace_stats.py",
        description="Compute scheduling statistics from an NNHaul"
                    " Chrome Trace Event dump (format_version 1).",
        epilog=(
            "method notes:\n"
            "  wall span      metadata.wall_clock_span_us (the dumper\n"
            "                 writes last X ts - first X ts)\n"
            "  utilization    (union of the lane's interval set) / wall\n"
            "                 span, per lane of metadata.lanes\n"
            "  bytes          sum of the lane's X events' args.bytes -\n"
            "                 the copy events' one-chunk-class quantum -\n"
            "                 i.e. the direction's copied bytes on a\n"
            "                 chan lane\n"
            "  bubbles        maximal gaps strictly between consecutive\n"
            "                 intervals of an EXEC lane (a trailing gap\n"
            "                 is not a bubble: no work is outstanding\n"
            "                 after the last op)\n"
            "  ops            counts of the X events' args.kind"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", help="path to the trace JSON file")
    parser.add_argument("--json", action="store_true",
                        help="emit a stable machine-readable JSON summary"
                             " (raw units: us, bytes, fractions 0..1)"
                             " instead of the human report")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data = load_trace(args.trace)
    stats = compute_stats(data)
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_human(args.trace, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
