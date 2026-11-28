<!--
@copyright (c) 2026-present Skolkovo Institute of Science and Technology
                            (Skoltech), Russia. All rights reserved.

NNHaul is a task-based dynamic execution engine over N memory nodes:
a virtual planner builds one plan per batch from a drained snapshot,
and a literal executor issues exactly the planned operations on the
lanes of the node inventory.

@file README.md
Repository overview: purpose, architecture, build and installation,
usage, environment knobs, tests, and the documentation index.

@version 2.0.0
-->

<!-- SPDX-License-Identifier: MIT -->

# NNHaul

Task-based dynamic execution engine — a C++17 **chunk-graph
multi-device executor** over k CUDA device nodes plus one host node.

> NNTile 2.0 pins `ncuda=1` (one CUDA device per process). The multi-device
> executor described below is a standalone NNHaul capability and is not used
> by the NNTile product.

## General purpose

NNHaul executes task graphs on **an inventory of memory nodes**: k
CUDA devices plus the host. Tasks are submitted in topological order
with an explicit node assignment, and nothing launches at submit —
submission is fast bookkeeping onto a pending chain. When the producer
closes a batch (`signal_graph_end()`), the **LRU lowering planner**
builds ONE plan of exactly four op kinds — `LINK`, `UNLINK`, `COPY`,
`TILE_OP` — over the per-node chunk pools, from a binding-table seed;
the **stateless plan checker** verifies the plan by one full fold
(fault names verbatim, a census on failure, nothing serviced); and the
**literal executor** issues exactly the planned operations on the
lane set — one execution lane per node plus one transfer lane per
connected node pair. `planned == issued` is a run invariant and a
witness in every trace.

Every tile decomposes, by the **one runtime chunk class**, into
`ceil(nbytes / chunk_class_bytes())` whole-class chunks (a runtime
power of two, default 4 MiB, initialized from `NNHAUL_CHUNK_SIZE_MB`); a `COPY` moves exactly its chunk's
occupied span — byte-exact, tail chunks move only their payload.
Residency is the ONE binding table: per node, chunk → (logical chunk,
actual/outdated). When a node's free list runs out the planner evicts
LRU (the victim order is fixed), rescues the last actual
replica of a live tile, and stages two-hop copies through the host
when no direct device channel exists.

Pipelining is depth-2: `signal_graph_end()` issues a batch without
waiting, `join_predecessor()` joins the older of two in-flight
batches, `wait()` is the wholesale barrier. The plan format and the
lane model are summarised above; the trace format is documented in
[docs/traces.md](docs/traces.md).

`nnhaul::cuda_stream()`, `nnhaul::cublas_handle()`,
`nnhaul::workspace_ptr()` and
`nnhaul::device_index()` are valid inside a codelet body and name the
RUNNING node. CPU kernels are first-class TILE_OPs on the host node: a
two-body codelet's TILE_OP node selects the body (the host node
dispatches the CPU body, a device node that device's CUDA body). The
`Kind` enum is a diagnostic label only. Worker ids ARE node ids —
devices 0..k-1, host k (`nnhaul::host_worker()`); `insert()` with
`worker_unspecified` resolves to the device-class default for a CUDA body
(node 0 unless `set_default_cuda_worker` changed it) and to the host node
for a CPU-only body.
There is no MPI, no distributed layer, no disk spill.

## Use cases

- **The NNTile 2.0 runtime.** NNTile links `libnnhaul` as its only
  backend; the NNTile core/tile/tensor/kernel suites
  and the kernel-vs-wrapper comparisons backed by the core suites run
  on it (the chunked classic-stack kernel decomposition is described in
  [docs/chunk-model.md](docs/chunk-model.md); the NNTile-side
  decomposition is documented in the NNTile backend docs).
- **Training with a VRAM budget.** Under `NNHAUL_LIMIT_CUDA_MEM` the
  planner spills whole class chunks to the host pool and refetches
  them batch by batch — transparently, as planned COPYs.
- **One runtime over several GPUs.** `init(0, k, ...)` with
  `NNHAUL_DEVICES` selects k devices; peer-probed channels carry
  device-to-device COPYs directly, and pairs without a hardware path
  stage through the host.
- **Pipelined batch execution.** `signal_graph_end()` closes and
  issues a batch without waiting, so the next batch's host-side graph
  build overlaps this batch's device work (depth 2).
- **Research substrate for scheduling and eviction policy.** The
  plan / check / execute separation makes every placement decision a
  printable, checkable artifact (`NNHAUL_PLAN_DUMP`); a lane-clock
  simulator over a user-loaded duration model predicts makespans
  (`nnhaul::set_duration` / `set_channel_rate`); and the tracer 1.0
  timeline joins measured against simulated ops
  ([docs/traces.md](docs/traces.md)).

## A minimal example

```cpp
#include <nnhaul/nnhaul.hh>

// A codelet carries its kernel bodies; the CUDA body runs on a
// device node's execution lane, where cuda_stream() names that node.
void write_val(void **buffers, void *cl_args) noexcept
{
    auto *v = static_cast<int *>(cl_args);
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    cudaMemcpyAsync(buf->get_ptr<int>(), v, sizeof(int),
        cudaMemcpyHostToDevice, nnhaul::cuda_stream());
}

void read_val(void **buffers, void *cl_args) noexcept
{
    auto *out = *static_cast<int **>(cl_args);
    auto *buf = static_cast<nnhaul::Buffer *>(buffers[0]);
    cudaMemcpyAsync(out, buf->get_ptr<int>(), sizeof(int),
        cudaMemcpyDeviceToHost, nnhaul::cuda_stream());
}

int main()
{
    // k = 1 device node (plus the host node); pools sized per the
    // knob precedence below. cl_args are copied at submit.
    nnhaul::init(0, 1, 1u << 20, 0);
    {
        nnhaul::Handle h(sizeof(int) * 1024); // one chunk (<= class)
        h.set_name("demo");
        int val = 42, out = 0;
        int *out_ptr = &out;
        nnhaul::Codelet writer("write_val", nullptr, write_val, nullptr);
        nnhaul::Codelet reader("read_val", nullptr, read_val, nullptr);
        // Submission order is the plan's queue order (topological).
        nnhaul::insert(writer, {{&h, nnhaul::Access::Write}},
            &val, sizeof(val));
        nnhaul::insert(reader, {{&h, nnhaul::Access::Read}},
            &out_ptr, sizeof(out_ptr));
        nnhaul::wait(); // out == 42
    } // Handle dtor retires the tile (it joins pending work first)
    nnhaul::shutdown();
}
```

Host-side reads and writes go through `handle.acquire(access)` — an
RAII `Acquired` pin of the tile's host-node bytes. A read acquire is
LINK plus per-chunk D2H COPYs into the pinned run (no host TILE_OP for a
contiguous gather); a Write acquire publishes through a host-node
TILE_OP at release. A
codelet with only a CPU body runs on the host node's lane
(`worker_unspecified` resolves it there); `NNHAUL_SIGNAL_GRAPH_END`
gates the pipelined close call for embedders. The full API is
documented in Doxygen comments in
[include/nnhaul/nnhaul.hh](include/nnhaul/nnhaul.hh).

## Build and installation

Requirements: CMake ≥ 3.18, a C++17 compiler, the CUDA toolkit
(compiler and runtime).

### Building on a CUDA host

```bash
cmake -S . -B build -GNinja
cmake --build build
ctest --test-dir build --output-on-failure
```

The default arch is `sm_75`; pass `-DCMAKE_CUDA_ARCHITECTURES=...` to
target another one. A container works
too (any image with the toolkit, CMake and Ninja); NNTile supplies
its own images for the parent build.

### Standalone build and install

Prerequisites on `PATH`: `cmake` (≥ 3.18), `ninja`, a C++17 `g++`,
and `nvcc` from a CUDA toolkit (12.x). The build needs a CUDA GPU at
test time only.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja
cmake --build build
ctest --test-dir build --output-on-failure   # 28 C++ + 2 Python tests
cmake --install build --prefix /opt/nnhaul   # or --prefix /usr/local
```

If the toolkit or compilers are not on `PATH`, point CMake at them
(adjust to your installation):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
    -DCUDAToolkit_ROOT=/usr/local/cuda \
```

The GPU architecture defaults to `sm_75`; pass
`-DCMAKE_CUDA_ARCHITECTURES=<list>` to target other devices. The chunk class is a RUNTIME value, read on first use from
`NNHAUL_CHUNK_SIZE_MB` (a power of two, default 4); the configure-time
option `-DNNHAUL_DEFAULT_CHUNK_SIZE_MB=4` fixes only the default.

A standalone install delivers `libnnhaul.a`; the copy embedded in
NNTile is shared (`libnnhaul.so`) so both libraries carry one engine.
It ships the headers under `include/nnhaul/` and a CMake package under
`lib/cmake/nnhaul/`
(`nnhaulConfig.cmake`, exported target `nnhaul::nnhaul`).

### Consuming an installed NNHaul

```cmake
cmake_minimum_required(VERSION 3.18)
project(myapp LANGUAGES CXX)
find_package(nnhaul 1.0 REQUIRED)   # -DCMAKE_PREFIX_PATH=/opt/nnhaul
add_executable(myapp myapp.cc)
target_link_libraries(myapp PRIVATE nnhaul::nnhaul)
```

`find_package(nnhaul)` re-finds Threads and the CUDA toolkit,
so the consumer needs no CUDA language enabled and no manual include
paths.

### Building NNHaul as a subproject

The same CMakeLists works through `add_subdirectory` / FetchContent
(the test suites default **off** in that mode):

```cmake
# Parent CMakeLists.txt (must enable CUDA, e.g. project(... CUDA CXX))
add_subdirectory(nnhaul)                 # tests off by default
# or, to skip install rules as well:
# add_subdirectory(nnhaul EXCLUDE_FROM_ALL)
target_link_libraries(myapp PRIVATE nnhaul::nnhaul)
```

With a plain checkout outside a parent tree, configure it standalone:

```bash
cmake -S nnhaul -B build-nnhaul -GNinja
cmake --build build-nnhaul
```

The build-tree alias is `nnhaul::nnhaul`; the plain target name
`nnhaul` is also visible. Set `NNHAUL_BUILD_TESTS=ON` to build the
test suites from inside the parent project.

### Using NNHaul from NNTile

NNHaul is vendored in the NNTile repository as `nnhaul/` and is the
default backend there: `cmake --preset nnhaul`. For a standalone NNTile
build against a separate NNHaul tree, configure that NNHaul tree with
`-DNNHAUL_LIBRARY_TYPE=SHARED` and pass
`-DNNHAUL_ROOT=<nnhaul build dir or install prefix>`; a static
`libnnhaul.a` is refused because it would give consumers a second engine. when the in-tree
subproject is used, NNTile's `$<BUILD_INTERFACE:nnhaul>` link resolves
to the subproject target directly.

## The plan and the pools

Placement is a planning act, not a runtime reaction. At close, the
planner walks the batch's tasks in submit order and, per operand
chunk, ensures a binding with ACTUAL data on the task's node — a COPY
from the lowest-NodeId actual source when the binding is outdated,
host-staged in two hops when no direct channel exists, a bare LINK
for a Write. Eviction on an exhausted free list is LRU by last
planned touch (the order is fixed), with a data-survival rescue (the
last actual replica of a live tile never dies bare), dying-victim
frees (dead tiles freed at their last use), and slack-gated early
refetch fires (the batch's own prefetch COPYs). Pool exhaustion is a
plan-time census fault, never serviced.

Every pool follows the **memory manager pool rule**: it starts EMPTY
(no preallocated memory without an explicit ask) and grows during the
compilation phase — through the two-phase ladder below, so a chunk's
address is stable for the run's life and a multi-chunk acquire or
merged driver copy stays inside one allocation — until a wall: its soft limit (`NNHAUL_LIMIT_CUDA_MEM` /
`NNHAUL_LIMIT_CPU_MEM`, or the `init()` cap arguments) reached, or
the wall of the NON-ALLOCABLE mode (below). **Growth acts follow the
two-phase ladder**: every allocation event tries TILE-WISE first (one
allocation for the whole demanding tile, contiguous by construction)
and falls back to single chunks for that event only on failure —
success by either path stays allocable and tile-wise is tried again
next time; a failed SINGLE-chunk allocation flips the pool
non-allocable permanently (allocations off). A pool under a soft
limit takes the remaining budget as one more ladder event. All growth
happens at planning time; nothing is freed until the very end; the
steady run allocates nothing. Growth is a compilation-phase act
only; the run phase executes the plan as ordered. Per-device fixed
workspaces (the GEMM gather/scatter workspace via
`nnhaul::workspace_ptr()`, the cuBLAS workspace, copy descriptors)
are allocated at init beside the pools and never count against the
limit. Dependencies
and events are DERIVED (the owed derivation plus an event registry),
never plan fields; the engine never polls an event — completion is
the queue-close barrier over all lanes.

Timing is a fact, never an input: load a duration model
(`nnhaul::set_duration`, `nnhaul::set_channel_rate`) and the
lane-clock simulator predicts the plan's makespan
(`simulate_makespan_ns` in the diagnostics); the same folded owed
sets the executor issues drive both sides. The old `insert(...,
sim=true)` replay queue is deleted machinery and fails loudly.

### Environment knobs

Read once at `init()` unless noted. A malformed value is loud: a warning
plus a documented fallback where one exists, otherwise a thrown error at
init or at first use (never a silent substitution).

- `NNHAUL_DEVICES` — the CUDA ordinals behind device NodeIds 0..k-1
  (default: ordinals 0..k-1 ascending). A list naming a different
  count than `init()`'s k is a loud init error.
- `NNHAUL_LIMIT_CUDA_MEM` — per-device arena SOFT LIMIT in MiB (the
  pool rule: pools start empty and grow on demand — TILE-SIZED acts,
  one chunk at a time as the fallback — until this soft limit or a
  failed one-chunk act; after that wall the pool is a fixed budget,
  never another growth attempt). When set it wins absolutely;
  garbage, non-positive, and impossibly large values throw at init.
  Unset (and no `init()` cap argument): unlimited — the wall is the
  first failed `cudaMalloc`, and the engine sticks to whatever it
  accumulated.
- `NNHAUL_LIMIT_CPU_MEM` — the host (CPU-RAM) pool's SOFT LIMIT in
  MiB. Precedence: this knob, else the `init()` `cpu_cap_bytes`
  argument, else unlimited. A malformed or negative value warns once
  and falls back to 4096 MiB; `0` is one block.
- `NNHAUL_PREALLOC_CUDA_MEM` / `NNHAUL_PREALLOC_CPU_MEM` — explicit
  PREALLOCATION head starts in MiB (per device / for the host pool):
  that much is allocated at INIT, each CHECKED against its limit (a
  head start above the limit is a loud init error, never clamped).
  NO preallocation by default (the rule); the growth episode still runs
  at the first demand, past the head start, to the wall.
- `NNHAUL_WORKSPACE_MB` — the per-device GEMM gather/scatter
  workspace (default 1024), allocated at init outside the chunk
  pools.
- `NNHAUL_CUBLAS_WORKSPACE_MB` — per-device cuBLAS workspace (default
  8; negative means no workspace; clamped to 4 GiB; a non-integer warns
  once and falls back to 8).
- `NNHAUL_EVENT_POOL` — exact per-device event provision (unset uses the
  formula: pow2ceil(2 * max outstanding per device), floored at 1024).

- `NNHAUL_CHUNK_SIZE_MB` — the RUNTIME chunk class (see the build
  section); a power of two, default 4. Initialized on first use by
  nnhaul; a malformed value is a loud configuration fault.
- `NNHAUL_TRACE` — the measured trace (see
  [docs/traces.md](docs/traces.md)): unset or `0` = off (the
  default); `1` dumps `nnhaul-trace.json` at `shutdown()`; any other
  non-empty value is the dump path verbatim. Mid-run dumps:
  `nnhaul::trace_dump(path)`.
- Diagnostics: `NNHAUL_PLAN_DUMP` (print every op of the named plan
  sequences at queue close — the placement analysis reads it
  offline), `NNHAUL_SYMBOL_TRACE`, `NNHAUL_GATE_TRACE`,
  `NNHAUL_ISSUE_BREAKDOWN`, `NNHAUL_ALLOC_TRACE`.
- RETIRED — a set knob warns once and is ignored: `NNHAUL_CHUNK_SIZES`,
  `NNHAUL_CHUNK_POLICY`, `NNHAUL_CHUNK_MIN_COUNT` (the chunk class is a
  runtime value read from `NNHAUL_CHUNK_SIZE_MB`),
  `NNHAUL_PLAN_CHECK` (one check, always full), `NNHAUL_BATCH_FENCE`,
  and — deleted machinery of the previous single-GPU engine —
  `NNHAUL_OFFLOAD_POLICY` and `NNHAUL_LAUNCH_AHEAD`.
  (`NNHAUL_CHUNK_TILES` is NOT an NNHaul knob: it is NNTile's
  tile-handle gate in `nntile/include/nntile/nnhaul/chunk_gate.hh`, where `=0` is
  rejected because this backend has no unchunked tiles.)

## Tracing

`NNHAUL_TRACE` records the run as one swimlane per lane of the node
inventory — `exec(node)` per node, `chan(a,b)` per connected ordered
pair — as Chrome Trace Event JSON (`format_version` 1) that loads
as-is into Perfetto (ui.perfetto.dev) and `chrome://tracing`, with
the plan witness (`planned == issued`) in the metadata. The lanes,
the format, the issue-tile clocks (what the timestamps mean — and
what they do not), the overhead (one host clock read
per instrumented op, zero CUDA API), and the analyzer and viewers are
documented in [docs/traces.md](docs/traces.md); a worked example sits
in [docs/examples/](docs/examples/).

## Tests

`ctest` in this repository runs the runtime suite (28 C++ and 2 Python
tests): the unit suites over the plan-engine types — host
pool, binding table (including a 10k-transition churn against a
reference fold), plan checker (the full fault table), dependency, the
timing pass (hand-derived lane arithmetic), the tracer, the executor
over real CUDA lanes (including the copy-run merge scenarios: merged
and unmerged realizations compared byte-for-byte), the LRU planner —
the engine suites (end-to-end smoke, depth-2 pipelining with the loud
third-batch abort, cross-batch survival/rescue round-trips, queue
admission, versioning, cap precedence, env-knob robustness, sgemm,
cpu_gpu_copy), and the trace pipeline (knob lifecycle, lane coverage
under pool pressure, plus two Python tests: the analyzer's numbers on
a hand-computed fixture and an end-to-end traced run cross-checked
against its sidecar). Multi-device and backend-conformance cases run
on real devices and skip loudly without them. When a `python3`
interpreter is not found, the two Python tests are not registered.

Tests that exercise NNTile on top of NNHaul — kernel-versus-wrapper
comparisons against the chunked classic stack — live in the NNTile tree
(`nntile/tests/nnhaul/`), not here. NNHaul's own suite has no NNTile
dependency.

## Documentation

- [docs/chunk-model.md](docs/chunk-model.md) — the chunk model: one
  chunk class, tile decomposition, tails, and what a codelet receives.
- [docs/planner-api.md](docs/planner-api.md) — the public planner
  interface and the example planner (`NNHAUL_PLANNER`).
- [docs/traces.md](docs/traces.md) — the trace manual: lanes, format,
  enabling, clocks, overhead, analyzer and viewers.
- [docs/examples/](docs/examples/) — a worked trace and its offline
  HTML report.

## Versioning

Every file header carries the release version in its `@version` tag, and
`project(nnhaul VERSION ...)` in `CMakeLists.txt` is the single source of
truth for it. Versioning follows semver: major for incompatible API
changes, minor for features, patch for fixes. A release updates the tag in
all files, `project(VERSION)`, and the `find_package` minimum when the
major version changes; the change lands as one commit tagged `vX.Y.Z`.

## Copyright

Copyright (c) 2026-present Skolkovo Institute of Science and
Technology (Skoltech), Russia. All rights reserved.

NNHaul is released under the MIT license; see [LICENSE](LICENSE).
