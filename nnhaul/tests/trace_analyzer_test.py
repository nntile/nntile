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
# @file tests/trace_analyzer_test.py
# Acceptance test for the trace analyzer: runs tools/trace_stats.py and
# tools/trace_check.py on tests/fixtures/trace_minimal.json (format
# version 1) and asserts every reported number equals the
# hand-computed expectation.
#
# @version 1.0.0

"""Acceptance test for the trace analyzer tooling (format_version 1).

Runs tools/trace_stats.py (JSON and human modes) and
tools/trace_check.py as subprocesses against
tests/fixtures/trace_minimal.json, asserting the exact --json schema
and every hand-computed number, plus error handling for bad input and
the checker's rejection of corrupted dumps. Standard library only;
exit status 0 on success.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "tools" / "trace_stats.py"
CHECKER = ROOT / "tools" / "trace_check.py"
FIXTURE = ROOT / "tests" / "fixtures" / "trace_minimal.json"

# Hand-computed expectations for tests/fixtures/trace_minimal.json
# (one device + host: lanes exec(0), exec(1), chan(0,1), chan(1,0)):
#
#   exec(0) TILE_OPs (tid 1): [0,40), [100,150), [150,210)
#     -> busy 150 us; exactly ONE internal gap [40,100) = 60 us (the
#        trailing gap after 210 is not a bubble: no work is left).
#   exec(1) TILE_OP (tid 2): [10,30) -> busy 20 us, no bubble.
#   chan(0,1) COPY (tid 3): [50,75) -> busy 25 us.
#   chan(1,0) COPY (tid 4): [20,50) -> busy 30 us.
#   span 150 us (metadata.wall_clock_span_us = the last X start
#     minus the first X start, 150 - 0; the dumper's own contract).
#   ops: 4 TILE_OP, 2 COPY; 6 X events total; 4 lanes.
#   bytes: both copies carry the dumper's one-chunk-class quantum
#     (4194304 = 4 MiB), so each chan lane's byte sum is exactly that
#     and the exec lanes' sums are 0 (tile ops carry no byte fact).

SPAN_US = 150
LANE_UTILIZATION = {"exec(0)": 150 / 150, "exec(1)": 20 / 150,
                    "chan(0,1)": 25 / 150, "chan(1,0)": 30 / 150}
LANE_BUSY_US = {"exec(0)": 150.0, "exec(1)": 20.0,
                "chan(0,1)": 25.0, "chan(1,0)": 30.0}
LANE_EVENTS = {"exec(0)": 3, "exec(1)": 1, "chan(0,1)": 1, "chan(1,0)": 1}
LANE_BYTES = {"exec(0)": 0, "exec(1)": 0,
              "chan(0,1)": 4194304, "chan(1,0)": 4194304}
BUBBLES = [{"lane": 0, "lane_name": "exec(0)", "start_us": 40,
            "end_us": 100, "dur_us": 60}]
OPS = {"copy": 2, "tile_op": 4}
PLAN = {"batches": 1, "planned_ops": 6, "issued_ops": 6, "fired_ops": 6,
        "reconcile_ok": True}


def run_tool(*args):
    return subprocess.run(
        [sys.executable, *[str(arg) for arg in args]],
        capture_output=True, text=True)


class AnalyzerJsonTest(unittest.TestCase):
    def setUp(self):
        proc = run_tool(ANALYZER, FIXTURE, "--json")
        self.assertEqual(proc.returncode, 0,
                         f"stderr: {proc.stderr}\nstdout: {proc.stdout}")
        self.stats = json.loads(proc.stdout)

    def test_schema_exact(self):
        stats = self.stats
        self.assertEqual(set(stats),
                         {"span_us", "lane_count", "x_events", "lanes",
                          "bubbles", "ops", "plan"})
        self.assertEqual(set(stats["ops"]), {"copy", "tile_op"})
        self.assertEqual(set(stats["plan"]),
                         {"batches", "planned_ops", "issued_ops",
                          "fired_ops", "reconcile_ok"})
        for lane in stats["lanes"]:
            self.assertEqual(set(lane),
                             {"lane", "tid", "name", "events", "busy_us",
                              "utilization", "bytes"})
        for bubble in stats["bubbles"]:
            self.assertEqual(set(bubble),
                             {"lane", "lane_name", "start_us", "end_us",
                              "dur_us"})

    def test_numbers_exact(self):
        stats = self.stats
        self.assertIsInstance(stats["span_us"], int)
        self.assertEqual(stats["span_us"], SPAN_US)
        self.assertEqual(stats["lane_count"], 4)
        self.assertEqual(stats["x_events"], 6)
        self.assertEqual({lane["name"] for lane in stats["lanes"]},
                         set(LANE_UTILIZATION))
        for lane in stats["lanes"]:
            name = lane["name"]
            self.assertEqual(lane["events"], LANE_EVENTS[name], name)
            self.assertEqual(lane["busy_us"], LANE_BUSY_US[name], name)
            self.assertEqual(lane["bytes"], LANE_BYTES[name], name)
            value = lane["utilization"]
            self.assertIsInstance(value, float)
            self.assertLessEqual(abs(value - LANE_UTILIZATION[name]), 1e-9,
                                 f"utilization[{name}]")

        self.assertEqual(stats["bubbles"], BUBBLES)
        self.assertEqual(stats["ops"], OPS)
        self.assertEqual(stats["plan"], PLAN)


class AnalyzerHumanTest(unittest.TestCase):
    def test_report_strings(self):
        proc = run_tool(ANALYZER, FIXTURE)
        self.assertEqual(proc.returncode, 0,
                         f"stderr: {proc.stderr}\nstdout: {proc.stdout}")
        for needle in ("Wall span", "Lane utilization", "exec(0)",
                       "exec(1)", "chan(0,1)", "chan(1,0)", "bubble",
                       "Ops by kind", "tile_op", "copy", "Plan witness",
                       "60.000", "reconcile_ok True"):
            self.assertIn(needle, proc.stdout)


class AnalyzerErrorTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_missing_file(self):
        proc = run_tool(ANALYZER, self.tmp / "no-such-trace.json")
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(proc.stderr.strip())

    def test_malformed_json(self):
        bad = self.tmp / "bad.json"
        bad.write_text("{not json at all", encoding="utf-8")
        proc = run_tool(ANALYZER, bad)
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(proc.stderr.strip())

    def test_no_arguments(self):
        proc = run_tool(ANALYZER)
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(proc.stdout.strip() or proc.stderr.strip())


class CheckerTest(unittest.TestCase):
    def test_fixture_ok(self):
        proc = run_tool(CHECKER, FIXTURE)
        self.assertEqual(proc.returncode, 0,
                         f"checker rejected the fixture: {proc.stderr}")
        self.assertIn("OK", proc.stdout)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def corrupt(self, name, mutate):
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        mutate(data)
        path = self.tmp / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def assert_rejected(self, path, needle=None):
        proc = run_tool(CHECKER, path)
        self.assertNotEqual(proc.returncode, 0,
                            f"checker accepted {path}: {proc.stdout}")
        self.assertTrue(proc.stderr.strip())
        if needle is not None:
            self.assertIn(needle, proc.stderr)

    def test_rejects_negative_dur(self):
        def mutate(data):
            for event in data["traceEvents"]:
                if event.get("ph") == "X" and event.get("tid") == 1:
                    event["dur"] = -5
                    return
        self.assert_rejected(self.corrupt("negative_dur.json", mutate),
                             "negative dur")

    def test_rejects_unknown_format_version(self):
        def mutate(data):
            data["metadata"]["format_version"] = 99
        self.assert_rejected(self.corrupt("unknown_format.json", mutate),
                             "format_version")

    def test_rejects_counter_events(self):
        def mutate(data):
            data["traceEvents"].append(
                {"ph": "C", "name": "vram_bytes", "pid": 1, "ts": 100,
                 "args": {"vram_bytes": 1048576}})
        self.assert_rejected(self.corrupt("counter_event.json", mutate),
                             "counter events")

    def test_rejects_missing_lane_args(self):
        def mutate(data):
            for event in data["traceEvents"]:
                if event.get("ph") == "X":
                    del event["args"]["lane_name"]
                    return
        self.assert_rejected(self.corrupt("no_lane_args.json", mutate),
                             "lane_name")

    def test_rejects_wrong_tid_for_lane(self):
        def mutate(data):
            for event in data["traceEvents"]:
                if event.get("ph") == "X" and \
                        (event.get("args") or {}).get("lane") == 2:
                    event["tid"] = 9
                    return
        self.assert_rejected(self.corrupt("wrong_tid.json", mutate),
                             "lane + 1")

    def test_rejects_non_monotone_ts(self):
        def mutate(data):
            kernels = [event for event in data["traceEvents"]
                       if event.get("ph") == "X" and event.get("tid") == 1]
            kernels[-1]["ts"] = 90  # previous op on tid 1 has ts 100
        self.assert_rejected(self.corrupt("non_monotone.json", mutate),
                             "non-decreasing")

    def test_rejects_broken_witness(self):
        def mutate(data):
            data["metadata"]["plan"]["issued_ops"] = 5
        self.assert_rejected(self.corrupt("broken_witness.json", mutate),
                             "planned_ops")

    def test_rejects_wrong_span(self):
        def mutate(data):
            data["metadata"]["wall_clock_span_us"] = 999
        self.assert_rejected(self.corrupt("wrong_span.json", mutate),
                             "wall_clock_span_us")

    def test_rejects_non_positive_bytes(self):
        def mutate(data):
            for event in data["traceEvents"]:
                if (event.get("args") or {}).get("kind") == "copy":
                    event["args"]["bytes"] = 0
                    return
        self.assert_rejected(self.corrupt("zero_bytes.json", mutate),
                             "positive number")


if __name__ == "__main__":
    unittest.main(verbosity=2)
