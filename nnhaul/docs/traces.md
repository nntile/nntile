<!--
@copyright (c) 2026-present Skolkovo Institute of Science and Technology
                            (Skoltech), Russia. All rights reserved.

NNHaul is a task-based dynamic execution engine over N memory nodes:
a virtual planner builds one plan per batch from a drained snapshot,
and a literal executor issues exactly the planned operations on the
lanes of the node inventory.

@file docs/traces.md
The measured trace of the chunk-graph multi-device executor: the
lane set of the node inventory rendered as Chrome Trace Event JSON,
enabling, the issue-tile clocks and what they measure (and what they
do not), the measured overhead, and the analyzer and viewers in
tools/.

@version 2.0.0
-->

<!-- SPDX-License-Identifier: MIT -->

# The trace

One environment knob turns an NNHaul run into a measured timeline of
the chunk-graph executor: one swimlane per lane of the run's node
inventory — an `exec(node)` lane per node, a `chan(a,b)` lane per
connected ordered node pair — written as a single JSON file in the
Chrome Trace Event format. The file is plain JSON that any script can
read, and the same bytes load unchanged into the Perfetto UI or
`chrome://tracing`, which draw the lanes — a professional swimlane
viewer for free, with no post-processing tool of our own to maintain.

The tracer lives in `src/plan_trace.hh/.cc` (armed in `src/engine.cc`
at init, dumped at shutdown); the rule it implements is
the issue-position stamp rule, with the simulated side beside
it in `src/timing.hh`.
This document is the user manual: the lanes, the format, how to
enable tracing, what the clocks measure (and what they do not), the
measured overhead, and the analyzer and viewers in `tools/`.

## The lanes

The frame is the LANE SET of the run's node inventory
(`NodeInventory`, built once at init from the node count and the
peer-probe channel matrix) — never a fixed list of streams, never a
vendor name:

- **`exec(0)` .. `exec(k-1)`** — one execution lane per CUDA device
  node, plus **`exec(k)`**, the host node's lane (the dedicated host
  worker's FIFO). A TILE_OP lands on the `exec` lane of the node it
  names.
- **`chan(a,b)`** — one transfer lane per connected ORDERED node
  pair. Device↔host pairs exist in both directions by construction;
  a device↔device pair exists iff the init-time peer probe said the
  hardware path is there. A COPY from node `a` to node `b` lands on
  `chan(a,b)` — the direction is the lane, so per-direction byte
  sums read straight off the trace.

On a one-device run (k = 1 device + host) the frame is four lanes:
`exec(0)`, `exec(1)`, `chan(0,1)` (D2H), `chan(1,0)` (H2D). The FULL
frame is the default and the dump carries every lane of the inventory
as a named thread even when a lane never carried an op — an empty
lane is itself a finding (a node or channel the planner left idle),
not an absence. Lane names come from `NodeInventory::lane_name` —
`exec(i)` / `chan(a,b)`, never vendors or runtimes.

LINK and UNLINK never appear: they are binding-table transitions in
engine memory at their issue instant — no lane, no device work, no
stamps (the stamp rule). They are visible in the plan, not the trace.

## The format

Chrome Trace Event JSON: one object with a `traceEvents` array and a
`metadata` block, pretty-printed (humans scan it, the tools parse
it). Intervals are `X` (complete) events — exactly one per
instrumented op. `pid` is always 1; `tid` is lane + 1; `ts` is
integer microseconds (whole-us floor against the first stamp, so the
emitted minimum is exactly 0); `dur` is fractional microseconds
written with three decimals — the exact nanosecond split, no
rounding (`3.257`, not `3`).

Two kinds of `M` metadata events open the array: one `process_name`
(`nnhaul`) and one `thread_name` per lane of the inventory, in lane
order — five events on the four-lane run. Every
`X` event names its op and carries its identity:

```json
{"ph": "X", "name": "tile_op 12.345", "cat": "exec", "pid": 1,
 "tid": 1, "ts": 109922, "dur": 12.204,
 "args": {"plan_seq": 12, "position": 345, "kind": "tile_op",
          "lane": 0, "lane_name": "exec(0)"}}
```

- `name` is `<kind> <plan_seq>.<position>` — the op's identity,
  readable at a glance; `cat` is `exec` or `chan`, matching the lane
  family.
- `args` carry `plan_seq` and `position` (the op's queue position
  within the plan) — the join key against the plan and the simulated
  side (the simulated records carry the same identity: slices keyed
  by `plan_seq`, `src/timing.hh`) — plus `kind`
  (`tile_op` or `copy`) and the lane fields (`lane` id beside its
  inventory `lane_name`) so the measured and simulated swimlanes can
  be diffed lane for lane.
- Copy events additionally carry `args.bytes` = the copy's real
  transferred bytes — the OCCUPIED SPAN of its source chunk
  (`tile_chunk_span`): the whole class (`nnhaul::chunk_class_bytes()`, 4194304
  by default — `NNHAUL_CHUNK_SIZE_MB=4`) for a full chunk, the payload
  remainder for a tail chunk. A direction's byte sum is the SUM of
  `args.bytes` over its copy events — not the event count times the
  constant (tail chunks move only their payload).

The `metadata` block carries `format_version` (1), `runtime_version`,
`lane_count`, `lanes` (the frame itself: `tid`, `lane`, `name` per
lane), `wall_clock_span_us` (last X `ts` minus first X `ts` — exact
for the checker), and the **plan witness**:

```json
"plan": {"batches": 1, "planned_ops": 4, "issued_ops": 4,
         "fired_ops": 4, "reconcile_ok": true}
```

`batches` is the plans registered; `planned_ops`/`issued_ops` are the
planned == issued reconciliation's two sides; `fired_ops` is the
completion-fact count — every instrumented op's stamp pair was read
complete at its plan's queue-close join; `reconcile_ok` is the AND of
the batches' reconciliation flags. A reader can verify planned ==
issued from the trace file alone. Within each lane `ts` is
monotonically non-decreasing in file order (one stable sort at dump;
equal `ts` keep issue order). There are no counter (`C`) events —
the previous engine's `vram_bytes`/`ram_pinned_bytes` tracks are
deleted machinery, and `tools/trace_check.py` rejects a dump that
carries them. Unknown extra keys are tolerated.

## Why Chrome Trace Event JSON

The format serves two consumers at once, which is the whole reason it
was chosen over a bespoke schema:

- **Any script.** The file is ordinary JSON — `json.load` and a
  comprehension over `traceEvents` is the entire integration cost.
  `tools/trace_stats.py` (below) is exactly that.
- **Professional viewers, as-is.** Open
  [ui.perfetto.dev](https://ui.perfetto.dev) and pick *"Open trace
  file"*, or load the file in `chrome://tracing`. The lanes render as
  named swimlanes (the `thread_name` metadata) and the intervals
  render with their names and `args` — no conversion step, no export
  filter, and the viewers' pan/zoom/search come for free.

NNHaul ships no GUI of its own for the raw timeline; the offline
report below covers the no-browser case.

## Enabling

`NNHAUL_TRACE` is read once at `init()` (per init cycle, like the
other knobs):

- unset, or `0`: tracing fully off (the default). Nothing is placed,
  nothing allocated, no clock read — an untraced run and a traced run
  share ONE issue discipline.
- `1`: on; the trace is dumped to `nnhaul-trace.json` in the process
  working directory at `shutdown()`.
- any other non-empty value: on; the value is the dump path verbatim
  (e.g. `NNHAUL_TRACE=/tmp/run.json`).
- an empty value: a stderr warning and tracing off (house style for
  garbage knobs — notice and a documented fallback, never a crash).

**When the dump happens.** At `shutdown()`, after the final join —
every un-retired batch's queue-close barriers have run, nothing can
append — the engine writes the file, then disarms the tracer. A
normal shutdown therefore produces the complete run. Mid-run, the
facts accumulate inside the tracer (per-plan harvests at the
queue-close joins) but no file is written until asked.

Two programmatic hooks in
[`include/nnhaul/test_hooks.hh`](../include/nnhaul/test_hooks.hh)
complete the picture:

- `bool nnhaul::trace_enabled()` — true while tracing is armed (a
  runtime exists and the knob enabled it at init).
- `bool nnhaul::trace_dump(std::string const &path)` — write what the
  trace has accumulated so far, in the same format, and **keep
  accumulating**; a later dump (or the `shutdown()` flush) contains
  everything plus the events since. Returns false when tracing is off,
  no runtime exists, or the write failed (the failure also reports to
  stderr). This is the mid-run window: dump, look, continue.

## Clocks, and what the timestamps mean

The clock rule is the issue-position stamp rule, in user
terms:

- **A stamp is one host-clock read** (`std::chrono::steady_clock`, ns
  domain) taken at the op's lane-ISSUE position — the instant the op
  is submitted into its lane. The submission into the lane IS in the
  lane's order by construction (one issue order per lane,
  non-blocking submission), so the read is AT the lane position
  whatever thread submits. The read is one host clock read and touches
  no CUDA API.
- **An op's measured interval is the tile it owns of its lane's issue
  order**: `[its stamp, the next stamp on that lane]`. One more read
  per touched lane at the issue's end — the plan's tail stamp —
  closes the last tile. On each lane the tiles chain: the op's tile
  ends where the next op's begins.
- **Durations INCLUDE the lane's queue wait** — stated plainly. A tile is
  the op's span of the lane's issue order,
  not the device work's own execution. The tile's START is the op's
  own submission; its END is the next read on the lane — the next
  op's SUBMISSION, which a non-blocking issue path can place before
  the op's device work has completed when the queue is deep (the
  lane's full serialization orders EXECUTION, not submissions). So
  the device work usually lies inside the tile, but a deep queue can
  push the tile's end before that work finishes. Read a tile as "how
  long this op was outstanding on its lane", not "how long the device
  spent on it".
- **Device-lane tiles are issue-order facts, not device-order
  windows.** The lane executes in issue order, so the tiles chain in
  execution order too, but a tile's bounds are host submission
  instants; the device-true per-op execution time is not in the file
  (below).
- **The host lane's tiles are exact.** `exec(k)` is one dedicated
  worker thread executing its FIFO in order, reading its own clock at
  each FIFO position — the worker IS the lane, so a host tile is the
  op's own execution span, exact by construction.
- **Stamps are facts, never inputs.** They gate, skip, and reorder
  nothing; tracing ON and OFF run the identical issue path. The facts
  are read only at the queue-close join, after the barriers — an
  instrumented position whose stamp pair did not land is a loud
  stamp-fidelity census, never a silent gap.

The dump's `ts` values are whole-us floors of the ns reads against
the first stamp; `dur` keeps the exact ns.

## What the trace does not contain

- **Device-true per-op execution durations.** The stamp pair brackets
  the op's slot in the lane's issue order; isolating the device work
  itself would need CUPTI activity records — the recorded upgrade
  path if a question ever demands it. No per-op CUDA insertion
  (timing events included) fits the timing budget (below), so the file
  trades device-side execution truth for a trace the loop cannot
  feel.
- **Device-side completion order.** The trace carries the completion
  FACT per op (the pair read at the join, counted in `fired_ops`),
  not the order completions arrived in device time.
- **LINK and UNLINK.** No lane, no stamps — they exist in the plan
  and the issue log, not the trace.

An empty stretch on a lane means no op was submitted to it in that
stretch — with the issue-tile clock there is no anchoring error to
discount, but there is also no device-side traffic in the file that
the issue path did not name.

## Partial traces

The `NNHAUL_TRACE` dump happens in `shutdown()`, after the final join
— a normal shutdown produces the complete run. If the process dies
mid-run, nothing is written at kill time; only whatever an earlier
programmatic `trace_dump()` wrote survives. For runs whose death you
want to autopsy, call `trace_dump()` at checkpoints.

## Overhead

Off (the default), tracing costs one armed-check branch per
instrumentation point — no CUDA calls, no allocations, no clock
reads; the issue discipline, stream ordering, and hot-path cost are
identical to the traced run's.

On, the per-op cost is the issue-tile read: **one host clock read per
instrumented op at its lane's issue position, zero CUDA API** — the submission path's own work, plus the JSON write at dump
time.

Two other mechanisms were evaluated and rejected before the issue
tile:


- **Stream-ordered host functions as stamps.** A host function
  inserted into a device stream stalls that stream while the driver
  dispatches and runs it, stalling that stream for the duration. The
  approach needs three placements per instrumented op, which is enough to
  double a loop's wall time.
- **Timing-event records as stamps.** A CUDA timing-event record
  stalls no device work, but it is not free on the HOST side: the
  record's submission cost grows with the stream's pending depth, while
  the device-side cost stays small. Two records per op visibly grow an
  iteration's wall time, measurably.

Either per-op CUDA insertion taxes a path the timing budget cannot
spare; the issue-position clock read taxes neither.

## Analyzing a trace

```bash
python3 tools/trace_check.py <trace.json>          # invariants (exit 0 iff valid)
python3 tools/trace_stats.py <trace.json>          # human report
python3 tools/trace_stats.py <trace.json> --json   # machine summary
```

`tools/trace_check.py` validates the format contract on any dump:
metadata keys and the format version, the per-lane `thread_name`
frame against `metadata.lanes`, the X events' fields and lane args
(`tid` = lane + 1, `cat` matching the lane family, `name` matching
the args' identity), per-lane `ts` monotonicity in file order, the
`wall_clock_span_us` equality, the plan witness's planned == issued
under `reconcile_ok`, and the absence of the deleted counter events.

`tools/trace_stats.py` parses any valid dump and reports the wall
span, per-lane utilization (busy fraction of the wall span) and event
counts over the lane set, the exec lanes' internal idle bubbles (a
maximal gap strictly between consecutive tiles while later ops still
exist — a trailing gap is idle, not a bubble), op counts by kind, the
per-direction copy byte sums read off `args.bytes`, and the plan
witness echo. The `--json` mode emits the same numbers as a stable
machine-readable summary (raw units: us, bytes, fractions 0..1),
asserted number-for-number by `tests/trace_analyzer_test.py`.

## Viewing a trace

- **Offline, no browser:**
  ```bash
  python3 tools/trace_report.py <trace.json> -o trace.html
  ```
  writes a single self-contained HTML file — one swimlane row per
  lane of the inventory over a shared time axis (inline SVG, idle
  lanes rendered as the findings they are), the plan witness in the
  header. No network, no CDN, no scripts; it renders straight from
  `file://` and is byte-identical across regenerations of the same
  trace.
- **Perfetto:** open <https://ui.perfetto.dev>, click *"Open trace
  file"*, and pick the `.json` — or drag it onto the page.
  `chrome://tracing` (Load button) reads the same file.
- **A worked example:** [docs/examples/trace.html](examples/trace.html)
  is a self-contained offline report of
  [docs/examples/nnhaul-demo.json](examples/nnhaul-demo.json) — four
  lanes (`exec(0)`, `exec(1)`, `chan(0,1)`, `chan(1,0)`), all populated,
  with the plan witness `planned == issued` in the metadata.

## Tests

- `trace_knobs` (`tests/test_trace_knobs.cc`) — the knob lifecycle,
  one init/workload/shutdown cycle per value: unset/`0`/empty leave
  no file, `1` dumps to the CWD at shutdown, a custom path is honored
  verbatim, mid-run `trace_dump()` output is parseable and grows with
  real batches; structural checks assert the four inventory lanes
  named, the format-1 metadata and plan witness present, and the old
  machinery (C counters, vram/ram keys, the three-stream lane names)
  absent.
- `trace_lanes` (`tests/test_trace_lanes.cc`) — interval capture
  under real pool pressure: a 12-tile working set against a
  4-chunk-class cap forces staged host hops and evictions, and the
  dump must show all four lanes carrying intervals with the
  planned == issued witness; it also writes a stats sidecar of the
  `test_hooks` counters for the pipeline test to cross-check.
- `trace_analyzer` (`tests/trace_analyzer_test.py`; registered in
  `ctest` when a `python3` interpreter is found, not registered
  otherwise) — `trace_stats.py` and `trace_check.py` against the
  hand-computed fixture `tests/fixtures/trace_minimal.json`: every
  `--json` number asserted (utilizations, busy times, the one-bubble
  definition, byte sums), the `--json` schema pinned, error handling
  and the checker's rejection of corrupted dumps covered.
- `trace_pipeline` (`tests/trace_pipeline_test.py`) — the end-to-end
  pipeline: runs the `trace_lanes` binary with tracing kept,
  validates its dump with `trace_check.py`, and cross-checks the
  trace against the workload's sidecar (op counts exact, plan
  witness exact, per-lane busy durations within rounding of the lane
  counters) and against `trace_stats.py --json`.
- `plan_trace` (`tests/test_plan_trace.cc`) — the tracer's unit
  suite, pure CPU: `knob_path()` semantics; the facts-at-the-join
  flow (the harvest composes the measured records from the stamp
  pairs, durations end-minus-start, ordered within a lane, completion
  facts accounted); the swimlane framing (the full frame renders
  empty lanes as findings, touched-so-far a monotone union); the join
  against the simulated trace (lane-for-lane equality over
  (plan_seq, position), mismatches detected, both sides' times
  exposed as facts and never compared); the loud belts (a COPY with
  no channel, a repeated plan_seq, an unknown or double harvest, a
  short fact vector, stamp fidelity); and the JSON dump round-trip
  over a temp file (every ts/dur derived from the stamps, per-lane ts
  non-decreasing, the lane fields on every event and in the
  thread_name metadata for every lane, dumps accumulating).
