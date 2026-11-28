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
# @file tools/trace_check.py
# Invariant validator for NNHaul Chrome-Trace dumps (format_version 1):
# checks the format contract — metadata keys, the per-lane thread_name
# frame, the X events' fields and lane args, per-lane ts monotonicity
# in file order, the wall-clock span equality, the plan witness, and
# the absence of the deleted counter events. Exit 0 iff every
# invariant holds.
#
# @version 2.0.0

"""Validate an NNHaul Chrome Trace Event dump (format_version 1).

Usage: python3 tools/trace_check.py <trace.json>

Exit status is 0 iff the trace parses and satisfies every invariant of
the NNHaul trace format contract; otherwise numbered violations are
printed to stderr and the exit status is non-zero. Unknown extra keys
and events (other than the counter events the old format carried —
those are violations) are tolerated.
"""

import json
import sys

FORMAT_VERSION = 1
REQUIRED_METADATA = (
    "format_version",
    "runtime_version",
    "lane_count",
    "lanes",
    "wall_clock_span_us",
    "plan",
)
PLAN_KEYS = ("batches", "planned_ops", "issued_ops", "fired_ops",
             "reconcile_ok")
X_FIELDS = ("ph", "name", "cat", "pid", "tid", "ts", "dur", "args")
X_ARG_FIELDS = ("plan_seq", "position", "kind", "lane", "lane_name")
KINDS = ("copy", "tile_op")


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check(data):
    """Return the list of contract violations (empty when valid)."""
    violations = []
    events = data.get("traceEvents")
    if not isinstance(events, list):
        return ["'traceEvents' is missing or not an array"]

    meta = data.get("metadata")
    if not isinstance(meta, dict):
        violations.append("'metadata' is missing or not an object")
        meta = {}
    for key in REQUIRED_METADATA:
        if key not in meta:
            violations.append(f"metadata is missing '{key}'")

    # The format identity.
    if meta.get("format_version") != FORMAT_VERSION:
        violations.append(
            f"format_version is {meta.get('format_version')!r}, expected"
            f" {FORMAT_VERSION}")

    # The lane frame: one metadata entry per lane of the inventory,
    # dense lane ids, unique tids.
    lanes = meta.get("lanes")
    lane_name = {}   # lane id -> name
    tid_lane = {}    # tid -> lane id
    if isinstance(lanes, list) and lanes:
        for index, entry in enumerate(lanes):
            if not isinstance(entry, dict):
                violations.append(f"metadata.lanes[{index}] is not an"
                                  " object")
                continue
            lane, tid, name = entry.get("lane"), entry.get("tid"), \
                entry.get("name")
            if not is_number(lane) or not is_number(tid) or \
                    not isinstance(name, str):
                violations.append(
                    f"metadata.lanes[{index}] needs numeric 'lane'/'tid'"
                    f" and a string 'name'")
                continue
            if lane in lane_name:
                violations.append(f"lane {lane} appears twice in"
                                  " metadata.lanes")
            if tid in tid_lane:
                violations.append(f"tid {tid} appears twice in"
                                  " metadata.lanes")
            lane_name[int(lane)] = name
            tid_lane[int(tid)] = int(lane)
        if sorted(lane_name) != list(range(len(lane_name))):
            violations.append("metadata.lanes ids are not dense"
                              " 0..lane_count-1")
        if is_number(meta.get("lane_count")) and \
                int(meta["lane_count"]) != len(lane_name):
            violations.append(
                f"metadata.lane_count {meta['lane_count']} != len("
                f"metadata.lanes) {len(lane_name)}")

    named_lanes = set()
    last_ts = {}
    x_ts = []
    x_events = 0
    kind_counts = {}

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            violations.append(f"traceEvents[{index}] is not an object")
            continue
        phase = event.get("ph")
        if phase == "M":
            if event.get("name") == "thread_name" and is_number(
                    event.get("tid")):
                tid = int(event["tid"])
                if tid in named_lanes:
                    violations.append(
                        f"traceEvents[{index}]: duplicate thread_name"
                        f" for tid {tid}")
                named_lanes.add(tid)
                args = event.get("args")
                name = args.get("name") if isinstance(args, dict) else None
                if tid in tid_lane and name != lane_name[tid_lane[tid]]:
                    violations.append(
                        f"traceEvents[{index}]: thread_name for tid {tid}"
                        f" is {name!r}, metadata.lanes says"
                        f" {lane_name[tid_lane[tid]]!r}")
        elif phase == "X":
            where = f"traceEvents[{index}] ('{event.get('name')}')"
            x_events += 1
            for field in X_FIELDS:
                if field not in event:
                    violations.append(
                        f"{where}: X event is missing '{field}'")
            ts, dur = event.get("ts"), event.get("dur")
            if not is_number(ts):
                violations.append(f"{where}: 'ts' is missing or not a"
                                  " number")
            if not is_number(dur):
                violations.append(f"{where}: 'dur' is missing or not a"
                                  " number")
            if is_number(ts):
                if ts < 0:
                    violations.append(f"{where}: negative ts {ts}")
                x_ts.append(ts)
                tid = event.get("tid")
                if is_number(tid):
                    tid = int(tid)
                    if tid in last_ts and ts < last_ts[tid]:
                        violations.append(
                            f"{where}: ts {ts} < previous ts"
                            f" {last_ts[tid]} on tid {tid} (per-lane ts"
                            " must be non-decreasing in file order)")
                    last_ts[tid] = ts
                    if tid not in tid_lane:
                        violations.append(
                            f"{where}: tid {tid} has no lane in"
                            " metadata.lanes")
            if is_number(dur) and dur < 0:
                violations.append(f"{where}: negative dur {dur}")

            # The lane args: the lane fields the tools diff swimlanes
            # by, consistent with the metadata frame and the event's
            # own cat/name.
            args = event.get("args")
            if not isinstance(args, dict):
                violations.append(f"{where}: 'args' is missing or not an"
                                  " object")
                args = {}
            for field in X_ARG_FIELDS:
                if field not in args:
                    violations.append(
                        f"{where}: X args are missing '{field}'")
            kind = args.get("kind")
            if kind in KINDS:
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
            elif kind is not None:
                violations.append(f"{where}: unknown kind {kind!r}")
            # The copy quantum: a COPY moves exactly one whole chunk of
            # the one chunk class, and the dumper stamps that byte
            # count on every copy event so a direction's byte sum is
            # readable off the trace. Optional (older dumps and
            # hand fixtures may omit it); present, it must be a
            # positive number.
            if "bytes" in args:
                if not is_number(args["bytes"]) or args["bytes"] <= 0:
                    violations.append(
                        f"{where}: args.bytes {args['bytes']!r} is not a"
                        f" positive number")
            if is_number(args.get("lane")):
                lane = int(args["lane"])
                name_arg = args.get("lane_name")
                if lane not in lane_name:
                    violations.append(
                        f"{where}: lane {lane} has no metadata.lanes"
                        " entry")
                elif name_arg != lane_name[lane]:
                    violations.append(
                        f"{where}: lane_name {name_arg!r} disagrees with"
                        f" metadata.lanes ({lane_name[lane]!r})")
                if is_number(event.get("tid")):
                    if int(event["tid"]) != lane + 1:
                        violations.append(
                            f"{where}: tid {event['tid']} != lane + 1"
                            f" ({lane} + 1)")
                expected_cat = ("exec" if str(name_arg).startswith("exec(")
                                else "chan"
                                if str(name_arg).startswith("chan(")
                                else None)
                if expected_cat is not None and \
                        event.get("cat") != expected_cat:
                    violations.append(
                        f"{where}: cat {event.get('cat')!r} does not"
                        f" match the lane name {name_arg!r}")
                if kind is not None and isinstance(
                        args.get("plan_seq"), (int, float)) and \
                        isinstance(args.get("position"), (int, float)):
                    expected_name = f"{kind} {int(args['plan_seq'])}." \
                                    f"{int(args['position'])}"
                    if event.get("name") != expected_name:
                        violations.append(
                            f"{where}: name {event.get('name')!r} != the"
                            f" args' {expected_name!r}")
        elif phase == "C":
            # The counter events of the previous format are deleted
            # machinery; a dump carrying them is not format 2.
            violations.append(
                f"traceEvents[{index}]: counter events ('ph': 'C') do"
                " not exist in format 2")
        # Any other phase is an unknown extra event: tolerated.

    for tid in sorted(tid_lane):
        if tid not in named_lanes:
            violations.append(
                f"missing thread_name metadata event for tid {tid}"
                f" (lane {tid_lane[tid]}, {lane_name[tid_lane[tid]]!r})")
    for tid in sorted(set(named_lanes) - set(tid_lane)):
        violations.append(
            f"thread_name for tid {tid} names no lane of metadata.lanes")

    if x_events < 1:
        violations.append("no X events found (need at least one"
                          " instrumented op interval)")

    # The plan witness block.
    plan = meta.get("plan")
    if isinstance(plan, dict):
        for key in PLAN_KEYS:
            if key not in plan:
                violations.append(f"metadata.plan is missing '{key}'")
        if plan.get("reconcile_ok") is True and \
                plan.get("planned_ops") != plan.get("issued_ops"):
            violations.append(
                f"metadata.plan.reconcile_ok is true but planned_ops"
                f" {plan.get('planned_ops')} != issued_ops"
                f" {plan.get('issued_ops')}")
    elif "plan" in meta:
        violations.append("metadata.plan is not an object")

    # The dumper writes metadata.wall_clock_span_us as the last X ts
    # minus the first X ts (the M events are pinned at ts 0 and sit
    # outside the span). Assert that equality exactly.
    span = meta.get("wall_clock_span_us")
    if x_ts and is_number(span):
        derived = max(x_ts) - min(x_ts)
        if span != derived:
            violations.append(
                f"metadata.wall_clock_span_us {span} != max ts - min"
                f" ts over the X events ({derived})")
    elif not x_ts and is_number(span) and span != 0:
        violations.append(
            f"metadata.wall_clock_span_us {span} != 0 for an empty"
            " event timeline")
    return violations


def main(argv=None):
    if len(argv or sys.argv[1:]) != 1:
        print("usage: tools/trace_check.py <trace.json>", file=sys.stderr)
        return 2
    path = argv[0] if argv else sys.argv[1]
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        print(f"trace_check: error: cannot read {path}: {exc}",
              file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"trace_check: error: {path} is not valid JSON: {exc}",
              file=sys.stderr)
        return 1
    if not isinstance(data, dict):
        print("trace_check: error: top-level value is not a JSON object",
              file=sys.stderr)
        return 1
    violations = check(data)
    if violations:
        for number, message in enumerate(violations, 1):
            print(f"trace_check: {number}: {message}", file=sys.stderr)
        print(f"trace_check: FAIL: {len(violations)} violation(s) in"
              f" {path}", file=sys.stderr)
        return 1
    print(f"OK: {path} passes all trace invariants")
    return 0


if __name__ == "__main__":
    sys.exit(main())
