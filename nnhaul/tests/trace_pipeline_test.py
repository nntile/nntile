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
# @file tests/trace_pipeline_test.py
# End-to-end trace pipeline test: runs the trace_lanes workload
# binary, with tracing kept, validates its dump (format_version 1)
# with trace_check.py, and cross-checks the trace against the
# workload's test_hooks sidecar and the analyzer's --json summary.
#
# @version 1.0.0

"""End-to-end test of the trace pipeline over a real GPU workload.

Usage: python3 tests/trace_pipeline_test.py <lanes-binary> <source-root>

The workload binary (trace_lanes) sets its own NNHAUL_TRACE path, runs
a cap-pressured workload over the inventory's lanes, dumps
trace-lanes.json plus a trace-lanes.stats.json sidecar of the
test_hooks counters (batches, the planned == issued witness, copy and
tile op counts, per-lane busy ns), and exits nonzero on failure. This
script runs it with NNHAUL_TEST_KEEP_TRACE=1 in its own temporary
directory, validates the dump with tools/trace_check.py, and asserts
the trace agrees with its sidecar (op counts exact, the plan witness
exact, per-lane busy durations within rounding of the lane counters)
and with tools/trace_stats.py --json (utilizations in [0, 1], one lane
row per inventory lane). Standard library only; exit status 0 on
success, nonzero with a message otherwise.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

WORKLOADS = {"trace_lanes": "trace-lanes"}


def fail(message):
    print(f"trace_pipeline: FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def run(command, cwd=None, env=None):
    return subprocess.run([str(part) for part in command], cwd=cwd,
                          env=env, capture_output=True, text=True)


def load_json(path, what):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        fail(f"cannot read the {what} {path}: {exc}")
    except json.JSONDecodeError as exc:
        fail(f"the {what} {path} is not valid JSON: {exc}")


def lane_sums(events):
    """Per-lane interval count and summed dur (us), keyed by lane id
    (the X events' args.lane), plus op counts by kind."""
    sums = {}
    ops = {}
    for event in events:
        if event.get("ph") != "X":
            continue
        args = event.get("args") or {}
        lane = args.get("lane")
        if not isinstance(lane, int):
            continue
        entry = sums.setdefault(lane, {"count": 0, "dur_us": 0.0})
        entry["count"] += 1
        entry["dur_us"] += float(event.get("dur", 0))
        kind = args.get("kind")
        if isinstance(kind, str):
            ops[kind] = ops.get(kind, 0) + 1
    return sums, ops


def check_workload(workload, stem, root):
    """Run one workload binary and cross-check its trace end to end."""
    trace_name = f"{stem}.json"
    sidecar_name = f"{stem}.stats.json"
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        env = dict(os.environ, NNHAUL_TEST_KEEP_TRACE="1")
        proc = run([workload], cwd=tmp, env=env)
        if proc.returncode != 0:
            fail(f"{workload.name} exited {proc.returncode};"
                 f" stdout: {proc.stdout.strip()[-400:]};"
                 f" stderr: {proc.stderr.strip()[-400:]}")
        trace_path = tmp / trace_name
        sidecar_path = tmp / sidecar_name
        if not trace_path.is_file():
            if "skip: no CUDA device" in proc.stdout:
                print("trace_pipeline: skip: no CUDA device")
                sys.exit(0)
            fail(f"{workload.name} produced no {trace_name} in its"
                 " working directory")

        # 1. The invariant validator accepts the dump.
        check = run([sys.executable, root / "tools" / "trace_check.py",
                     trace_path])
        if check.returncode != 0:
            fail(f"trace_check rejected the dump: {check.stderr.strip()}")

        trace = load_json(trace_path, "trace dump")
        sidecar = load_json(sidecar_path, "stats sidecar")
        sums, ops = lane_sums(trace["traceEvents"])

        # 2. The plan witness. The tracer registers EVERY plan of the
        # run (the close-and-plan batches the manager counts plus the
        # engine's inline acquire/publish plans), so its batches and
        # op counts bound the manager's from above; the witness itself
        # (planned == issued, reconcile) must hold exactly.
        meta_plan = trace["metadata"]["plan"]
        if meta_plan["batches"] < sidecar["batches"]:
            fail(f"metadata.plan.batches {meta_plan['batches']} < sidecar"
                 f" batches {sidecar['batches']}")
        if meta_plan["planned_ops"] < sidecar["planned_ops"] or \
                meta_plan["issued_ops"] < sidecar["issued_ops"]:
            fail(f"metadata.plan ops {meta_plan['planned_ops']}/"
                 f"{meta_plan['issued_ops']} below the sidecar's manager"
                 f" counts {sidecar['planned_ops']}/"
                 f"{sidecar['issued_ops']}")
        if meta_plan["planned_ops"] != meta_plan["issued_ops"]:
            fail(f"metadata.plan planned {meta_plan['planned_ops']} !="
                 f" issued {meta_plan['issued_ops']}")
        if meta_plan["reconcile_ok"] is not (sidecar["reconcile_ok"] is
                                             True):
            fail(f"metadata.plan.reconcile_ok"
                 f" {meta_plan['reconcile_ok']} != sidecar"
                 f" {sidecar['reconcile_ok']}")
        if not meta_plan["reconcile_ok"]:
            fail("the workload's batches did not reconcile"
                 " (planned != issued)")

        # 3. Op counts: every issued COPY and TILE_OP carries exactly
        # one instrumented X event of its kind.
        if ops.get("copy", 0) != sidecar["copy_ops"]:
            fail(f"X copy events {ops.get('copy', 0)} != sidecar"
                 f" copy_ops {sidecar['copy_ops']}")
        if ops.get("tile_op", 0) != sidecar["tile_ops"]:
            fail(f"X tile_op events {ops.get('tile_op', 0)} != sidecar"
                 f" tile_ops {sidecar['tile_ops']}")

        # 4. Per-lane busy: the sidecar's lane_busy_ns (the traced
        # stamps' sums) equal the dump's summed durations per lane
        # (dur is ns-exact three-decimal us; allow one rounding step
        # per event).
        busy = sidecar["lane_busy_ns"]
        for lane, entry in sums.items():
            if lane >= len(busy):
                fail(f"X event on lane {lane} past the sidecar's"
                     f" lane_busy_ns ({len(busy)} lanes)")
            expect_us = busy[lane] / 1000.0
            slack = (entry["count"] + 1) * 1e-3
            if not expect_us - slack <= entry["dur_us"] <= \
                    expect_us + slack:
                fail(f"lane {lane}: summed dur {entry['dur_us']} us"
                     f" outside [{expect_us - slack:.3f},"
                     f" {expect_us + slack:.3f}] us from"
                     f" lane_busy_ns[{lane}]")
        for lane, busy_ns in enumerate(busy):
            if busy_ns > 0 and lane not in sums:
                fail(f"lane_busy_ns[{lane}] = {busy_ns} but the dump"
                     " carries no X event on that lane")

        # 5. The analyzer's summary: schema present and in range.
        stats_proc = run([sys.executable, root / "tools" / "trace_stats.py",
                          trace_path, "--json"])
        if stats_proc.returncode != 0:
            fail(f"trace_stats --json failed: {stats_proc.stderr.strip()}")
        stats = json.loads(stats_proc.stdout)
        lane_count = trace["metadata"]["lane_count"]
        if stats["lane_count"] != lane_count:
            fail(f"analyzer lane_count {stats['lane_count']} != metadata"
                 f" {lane_count}")
        for lane in stats["lanes"]:
            value = lane["utilization"]
            if not 0 <= value <= 1:
                fail(f"utilization[{lane['name']}] = {value}"
                     " outside [0, 1]")
        if "bubbles" not in stats:
            fail("analyzer summary has no 'bubbles' key")

        copy_count = ops.get("copy", 0)
        tile_count = ops.get("tile_op", 0)
        print(f"OK: {workload.name} pipeline holds - {lane_count} lanes,"
              f" {copy_count} copy / {tile_count} tile_op intervals,"
              f" batches {meta_plan['batches']}, planned =="
              f" issued {meta_plan['planned_ops']},"
              f" {len(stats['bubbles'])} exec bubbles")


def main(argv):
    if len(argv) != 3:
        print("usage: tests/trace_pipeline_test.py <lanes-binary>"
              " <source-root>", file=sys.stderr)
        return 2
    root = Path(argv[2]).resolve()
    workload = Path(argv[1]).resolve()
    if not workload.is_file():
        fail(f"workload binary not found: {workload}")
    # Target names are prefixed for in-tree builds (nnhaul_<test>).
    key = workload.name
    if key not in WORKLOADS and key.startswith("nnhaul_"):
        key = key[len("nnhaul_"):]
    if key not in WORKLOADS:
        fail(f"unknown workload binary {workload.name}; expected one"
             f" of: {', '.join(sorted(WORKLOADS))}")
    check_workload(workload, WORKLOADS[key], root)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
