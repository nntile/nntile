<!--
@copyright (c) 2026-present Skolkovo Institute of Science and Technology
                            (Skoltech), Russia. All rights reserved.

@file docs/planner-api.md
The public planner interface: what a planner reads, what it emits, how it
is loaded, and what an external planner may and may not assume.

@version 1.0.0
-->

<!-- SPDX-License-Identifier: MIT -->

# Planner API

`<nnhaul/planner.hh>` is the complete contract for a lowering planner.
A planner needs no other NNHaul header and no access to the engine's
private types. The in-tree LRU planner emits its plan through the same
`Sink`, so the output half of the interface is exercised on every run.
Its input half (`Topology`) is exercised by the example planner and the
plugin test; the LRU implementation still reads the engine's concrete
inventory internally (see "Limits").

## What a planner reads

`PlanRequest` carries one batch:

| Field | Meaning |
|---|---|
| `plan_seq` | run-scoped plan sequence number |
| `entry` | the entry residency fold: `(node, chunk) -> (tile, index, actual)`, sorted by `(node, chunk)` |
| `tasks` | the batch's tasks in submit order, each with its **execution node** (an input, never the planner's choice) and its operand tiles with access modes |
| `dead_tiles` | tiles retiring in this batch; their bindings are freed at their dying position |

`Topology` is the read-only world view:

| Method | Meaning |
|---|---|
| `node_count()`, `is_host()` | the inventory: devices `0..k-1`, host `k` |
| `pool_capacity(node)` | soft ceiling of the node's chunk pool, in chunks (`0` = unlimited) |
| `pool_chunks(node)`, `chunk_valid(node, chunk)` | the chunks the pool owns now; ids `0..pool_chunks-1` are valid |
| `has_channel(a, b)`, `channel_rate(a, b)` | direct transfer channel and its relative rate (bytes/ns) |
| `carries(tile)`, `tile_chunks(tile)`, `tile_bytes(tile)` | the tile decomposition and payload size |

## What a planner emits

`Sink` takes the plan's four operation kinds and two side facts:

```cpp
sink.link(tile, index, node, chunk);
sink.unlink(tile, index, node, chunk);
sink.copy(tile, index, {src_node, src_chunk}, {dst_node, dst_chunk});
sink.tile_op(task, node, operands);   // operands: tile, index, access, node, chunk
sink.grow(node, cardinality);         // pool growth during compilation
sink.set_exit(bindings);              // predicted exit residency
sink.set_stats(stats);                // optional counters
```

A COPY moves exactly the chunk's occupied span; the engine computes it.
`set_exit` must describe the residency after the whole queue. Its shape is
validated, but the next batch's `entry` and dead-tile retirement come from
the checker's exit fold, not from this prediction. Every emitted plan is
validated by the engine's checker before execution, exactly like the
in-tree planner's output.

## Loading

```
NNHAUL_PLANNER=/path/to/libmyplanner.so
```

The library exports two symbols:

```cpp
extern "C" std::uint32_t nnhaul_planner_api_version();   // planner::kApiVersion
extern "C" nnhaul::planner::Planner *nnhaul_create_planner();
```

A plugin that references engine symbols must link the shared
`libnnhaul`; a header-only planner that references none is accepted. When
the symbols are exported, the loader compares the identity token and the
actual `chunk_class` address and refuses a plugin that carries its own
engine. A plugin that hides those symbols (`-fvisibility=hidden`,
`--exclude-libs`) and any deliberately renamed/preloaded DSO are outside
that guarantee: an `LD_PRELOAD` can always run code in-process. A standalone static
NNHaul build has no separate shared engine to share with a plugin: configure
it with `-DNNHAUL_LIBRARY_TYPE=SHARED` if the plugin is loaded into a
process that also runs the engine.

The loader refuses a missing symbol, a version mismatch, or a null
planner, naming the expected and found versions. The planner object is
owned by the plugin and must outlive every batch (a process-lifetime
singleton is the expected shape). With `NNHAUL_PLANNER` unset, the in-tree
LRU planner runs.

## Writing one

`examples/planner/greedy_planner.cc` is a complete planner:
it keeps each needed chunk actual on the task's node, copies from any
actual replica when it is not, links a fresh chunk when no data exists,
and reports the exit residency. Build it, then:

```bash
cmake --build build --target nnhaul_example_greedy_planner
NNHAUL_PLANNER=build/greedy_planner.so ./your_program
```

The example is tested by `tests/test_planner_plugin.cc`, which loads it,
checks the negotiated API version, and plans a batch against a synthetic
topology without a GPU. It includes engine-internal headers, so it runs
from the source tree rather than against an installed SDK. `tests/test_engine_smoke.cc` is also run once more with
`NNHAUL_PLANNER` pointing at the example library, so a shipped plugin's
plan is checked by the engine, issued, and its result verified end to
end (ctest target `nnhaul_engine_external_planner`).

## Limits of v1

- One planner per process; it is selected once at engine initialization.
  The instance may serve several Runtimes, so `plan()` must be
  thread-safe.
- The in-tree LRU planner predates this interface: it consumes the
  concrete `NodeInventory`/`TileTable` and only its output goes through
  `Sink`. A third-party planner gets the real `Topology` accessors; the
  LRU's internal access path is a known gap, not a hidden privilege.
- `PlanStats` is diagnostic; a plugin may leave it zero.
- Pool growth is requested through `grow()`; a plan that needs chunks
  beyond the requested cardinalities fails the checker, not silently.
- `Topology` exposes pool cardinality, the soft limit, channels and
  rates, and the tile decomposition. It does **not** yet expose a
  per-task duration model, scratch/pin budgets, or a next-use query;
  a time-aware planner must derive what it can from the request and
  treat durations as its own input.
- Task placement is an input: `PlannedTask::node` is fixed at submit, so
  a planner chooses residency (copies, links, evictions), not which node
  runs a task. `Sink::tile_op` must carry that same node.
- A task occurrence is ONE `TILE_OP` with its full operand set: one
  entry per (submitted buffer, chunk), buffers in submission order and
  chunks ascending. The executor hands each buffer's run of `n`
  pointers straight to the codelet, so the engine's pre-issue check
  rejects a split, a permutation or a missing entry (the checker alone
  accepts splits). A task whose two
  buffers name one tile is refused at the next close-and-plan.
- A repeated TaskId may legitimately appear twice in the batch (two
  occurrences of one symbol); each occurrence still gets its own full
  `TILE_OP`, in submit order.
- `set_exit` is a PREDICTION. The engine validates its shape (nodes,
  tiles, chunks, duplicates) and then seeds the next batch and retires
  dead tiles from the checker's own fold, not from the prediction. A
  plugin whose exit disagrees with the fold still executes the plan it
  emitted, but its exit is not trusted for the next batch.
- The engine calls `plan()` while holding its runtime lock. A planner
  must not call back into the engine's public API from inside `plan()`.
