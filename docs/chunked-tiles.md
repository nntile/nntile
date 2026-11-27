<!-- SPDX-License-Identifier: MIT -->
<!--
@copyright (c) 2026-present Skolkovo Institute of Science and Technology
                            (Skoltech), Russia. All rights reserved.

NNHaul is a task-based dynamic execution engine over N memory nodes:
a virtual planner builds one plan per batch from a drained snapshot,
and a literal executor issues exactly the planned operations on the
lanes of the node inventory.

@file docs/chunked-tiles.md
User documentation: chunked tiles for classic NNTile ops — cache-chunk
storage, chunk-granular transfers, and the kernel decomposition that
keeps results identical to contiguous tiles (bit-exact for elementwise
ops, within each family's documented chaining tolerance for reductions).

@version 1.0.0
-->

# Chunked tiles for classic NNTile ops

Status: user documentation of the default (NNHaul) backend. It records
why chunking exists, the chunk model in user terms, and the NNTile
classic-stack kernel decomposition.
The requirement is **correctness**: every classic op must produce results
identical to the same op on a contiguous tile, proven by the test suites
run with chunking in force.

## 1. Why

Chunked offload attains its best throughput when transfers are
splittable: the two bus directions can then interleave at sub-tile
granularity and device space can be reclaimed fractionally. A
whole-tile residency — one handle, one contiguous device block, one
indivisible transfer of `nbytes` — can only realize whole-transfer
granularity. Storing tiles in **cache chunks** — whole-class blocks
drawn from the node's pool — makes the transfer unit the chunk, so the
planned COPYs and the LRU eviction of the chunk-graph executor become
chunk-granular.

Chunk size trades per-chunk latency against preemption granularity. The
earlier ladder-of-classes idea coupled that tradeoff with allocator
efficiency; the chunk-graph design collapsed it to **one runtime class
per process**
(`NNHAUL_CHUNK_SIZE_MB`, a runtime environment variable read on first
use by nnhaul, default 4 MiB, a power of two): the pools, the binding
table, and the kernels' chunk-index arithmetic all use a single
power-of-two class delivered as runtime mask/shift data, and the
chunk-size choice affects performance only, never results.

## 2. Model

A **tile** is a `Handle(nbytes)`:

- The tile's `nbytes` split into `n = ceil(nbytes / chunk_class_bytes())`
  chunks; every chunk occupies exactly one whole-class block of its
  node's pool, and the last chunk's logical payload may be shorter —
  the intra-chunk tail, wasted bytes inside the full class block (no
  tail chunk of any other size).
- Residency is **per chunk and per node**: the ONE binding table maps,
  on each node, the pool's blocks to the logical chunks that occupy
  them, with an actual/outdated flag per binding. A kernel use of the
  handle touches *all* its chunks; the planner moves chunks node to
  node as planned COPYs.
- A COPY moves exactly its chunk's **occupied span** —
  `min(class, nbytes − i·class)` bytes: full chunks move the whole
  class, a tail chunk moves only its payload remainder (byte-exact
  transfers).
- The host node's pool is pinned whole-class blocks, grown on demand
  in whole `cudaHostAlloc` segments under the pool rule
  (empty at init, growth at compilation only); host acquires are
  realized as LINK and per-chunk D2H COPY plan ops into the pinned host
  run; only a Write acquire publishes through a host-node TILE_OP at release
  (`Buffer::has_host_copy()` / `get_host_ptr()` for the single-chunk
  case).
- A tile at or below the class is the degenerate `n = 1` case:
  `Buffer::get_ptr<T>()` works exactly when `chunk_count() == 1` (it
  throws on a multi-chunk buffer, so a kernel can never silently read
  the first chunk as the whole tile), `get_chunk_ptr<T>(i)` addresses
  chunk i, and `Buffer::chain_contiguous()` reports whether the whole
  chunk chain sits at consecutive class-stride addresses — when it
  holds, the chain IS one flat classic buffer and a kernel taking a
  single plain pointer over the whole tile is bitwise the chunked
  execution (no byte staged).

### The one chunk class

The class is the **runtime value** `nnhaul::chunk_class_bytes()`,
initialized by nnhaul from `NNHAUL_CHUNK_SIZE_MB` (default 4, a power
of two) on first use — one class per process, fixed for every tile and
exposed to kernels as `chunk_class_mask()` and `chunk_class_shift()`:

| tile bytes `s` | decomposition |
|---|---|
| `s <= chunk class` | 1 chunk — one whole-class block, `get_ptr` valid, `chain_contiguous` trivially true |
| `s > chunk class` | `ceil(s / chunk class)` chunks of exactly the class (the last chunk's payload shorter; its block still whole-class) |

Rationale: large chunks keep per-copy bandwidth high and keep the
chunk-pointer table the kernels transport small (the graph's largest
transported tile is a large activation tile as many full chunks plus a tail), and the power-of-two class lets the kernels'
chunk-index divmod use shift/mask with the runtime shift and mask
delivered as kernel arguments. The power-of-two class keeps the general
64-bit integer-division expansion off the GPU's address path: chunk index
and in-chunk offset are a shift and a mask. There is **no interleaving floor
and no per-operand chunk-count cap**: a tile's count grows with its size,
subject to the launch-wide 2^20-slot transport, a 2^31-1-element
per-launch indexing envelope (the kernels' flat indices are 32-bit), and
the conv2d 4096-slot cap documented below.

The retired chunk-policy knobs — `NNHAUL_CHUNK_SIZES` (the former
ladder), `NNHAUL_CHUNK_MIN_COUNT`, and `NNHAUL_CHUNK_POLICY` — warn once
at init and are ignored; the former `NNHAUL_CHUNK_POLICY_LOG` debug hook
is gone with them. `NNHAUL_CHUNK_TILES` is not a knob of this backend: any
value is rejected (the 1.x `=0` opt-out produced contiguous tile handles).

## 3. Where chunked tiles live in the engine

The chunk model and the four-op protocol (LINK, UNLINK, COPY, TILE_OP)
are summarised in the README; in the sources:

- **`include/nnhaul/nnhaul.hh` / `src/api.cc`**: `Handle(nbytes)` —
  the two-argument constructor's `chunk_bytes` is accepted for source
  compatibility and ignored; `Buffer` carries `chunk_count()`,
  `chunk_bytes(i)`, `get_chunk_ptr<T>(i)`, the host-replica accessors,
  and `chain_contiguous()`.
- **`src/plan.hh`**: the chunk-graph contract types — `LogicalChunk`,
  `tile_chunk_count`, and `tile_chunk_span` (the occupied byte span
  every COPY realizes).
- **`src/binding_table.hh/.cc`**: the ONE binding table — per node,
  chunk id → (logical chunk, actual/outdated flag); the free lists are
  derived views of the table.
- **`src/lru_planner.hh/.cc`**: the lowering planner resolves each
  TILE_OP's operands chunk by chunk (a COPY from the lowest-NodeId
  actual source, host-staged two-hop when no direct channel exists)
  and evicts LRU when a node's free list runs out.
- **`src/plan_executor.hh/.cc`**: issues each COPY as one transfer of
  exactly the occupied span on the pair's channel lane.
- **`src/host_pool.hh/.cc`** grows the host node's pinned whole-class
  blocks in segments on demand (the pool rule); **`src/cuda_backend.hh/.cc`**
  grows each device's arena the same way, whole-class blocks with
  stable chunk addresses.

## 4. NNTile changes (classic, NNHaul only)

- **`core/tile.hh`** (NNHaul branch): the ctor allocates
  `Handle(storage_bytes)` — the runtime class fixes the decomposition
  (the legacy two-argument `Handle(nbytes, chunk_bytes)` form is
  accepted for compatibility and ignored).
- **`codelet.hh`**: chunk accessors for buffers.
- **`tensor/ops/gemm.cc`, `tile/ops/gemm.cc`**: the three GEMM operands
  must be distinct tensors/tiles on this backend (one TILE_OP carries one
  operand run per tile). Aliased GEMM such as `A @ Aᵀ` needs an explicit
  copy of the shared input.
- **Op wrappers (`src/nnhaul/*.cc`)** consume chunk tables:
  1. *Elementwise family* (add, fill, clear, copy, relu/gelu/silu
     families, scale, multiply, …): most have a dedicated chunked
     entry point (one launch over the chunk tables); where none exists
     the wrapper walks the chunk chain and calls the existing kernel
     per piece.
  2. *gemm*: an operand whose chunk chain is one contiguous run goes
     straight to cuBLAS. Every other operand is copied whole into the
     per-device GEMM workspace, the contiguous cuBLAS call runs there,
     and the result is copied back (the C2W/W2C staging in
     `nntile/src/nnhaul/gemm.cc`).
  3. *Unit-structured families* (shared `chunk_units.hh`: the
     `ChunkView`/`UnitSpan`/`walk_unit_runs` walker, `SliceStage`
     panel staging and gather/scatter arenas): native
     piecewise drivers that reproduce the plain whole-buffer call
     exactly for elementwise classes, within the per-family chaining
     tolerance for reductions, with zero staging. Promoted in four
     families: family 1, the softmax/slice class (11 ops:
     softmax(+inplace), maxsumexp, logsumexp, add_slice(+inplace),
     scale_slice, sum_slice, sumprod_slice, norm_slice(+inplace));
     family 2, the fiber/transpose class (10 ops, incl. the
     added `kernel::transpose::panel`); family 3, rope/subcopy/randn/
     mask_scalar/subtract_indexed_outputs/norm/sum/total_sum_accum
     (8 ops); and multiply_slice. Family 3's randn, mask_scalar,
     subtract_indexed_outputs and total_sum_accum also have dedicated
     chunked entry points; the others drive existing kernels piecewise.

  4. *Staging (removed)* — the former fallback gathered chunks into a
     contiguous scratch handle, ran the original kernel once and
     scattered back. No submit path in this tree takes it: the conv2d,
     swap_two_axes and transpose drivers carry chunked/panel kernels, and
     `stage.*` keeps the chunked-handle registry and scratch accounting
     alongside the historical staged helpers. The code comments retain the historical derivation for
     context.

### Dispatch modes

Every tile is chunk-backed, but a codelet may reach the device in one of three
ways:

1. **Dedicated chunked kernel** (62 families). One launch receives the chunk
   pointer tables and walks chunks by shift/mask; the result is independent of
   the chunk class.
2. **Chunk-aware driver** (transitional; no active family uses it now).
   The helper `for_each_flat_piece` walks the flat element range in pieces
   that each fit inside one chunk of every operand and calls the ordinary
   contiguous kernel once per piece. It remains for disabled families, CPU
   reference paths and future ops; every active family now has a one-launch
   chunked entry.
3. **Single-chunk fast path**. When every operand is exactly one chunk, the
   codelet may call the contiguous kernel directly. This is the only permitted
   contiguous launch: the buffer is still chunk-backed and the single chunk is
   the whole operand.

The one-launch chunked kernel is the preferred form. A per-chunk driver loop
issues one kernel launch per chunk; for a 1 MiB chunk class and a 256 MiB
tensor that is 256 sequential launches, which overloads the driver and limits
how far the CUDA async queue can run ahead. The driver mode is therefore a
transitional path for families that do not yet have a chunked entry; moving a
family to a one-launch chunked kernel removes that launch traffic. `gelu` was
converted this way: vs the per-chunk driver it is ~1.2x faster at 2 MiB and
~1.57x at 128–256 MiB with a 1 MiB chunk class.

### Gating

Chunked tile storage and dispatch exist only in the NNHaul stack. Every tile
of an NNHaul run is chunk-backed; dispatch follows the three modes above.
There is no engine-side gating knob: `NNHAUL_CHUNK_TILES` is NNTile's tile-handle
gate (`nntile/include/nntile/nnhaul/chunk_gate.hh`), it defaults to on and any
explicit value is rejected; every tile of an NNHaul run decomposes by the
runtime class; kernels that need one flat pointer over a multi-chunk
tile stage through the per-device workspace (`nnhaul::workspace_ptr()`,
notably GEMM) or exploit `chain_contiguous()` when the chain happens
to be flat.

### Chunk-pointer transport (the device table)

The one-launch chunked kernels need each operand's chunk base pointers
as a `T const* const*`. The transport is a **linear device table** of
chunk-pointer slots (`ChunkPtrTable`, `kChunkPtrTableSlots = 1 << 20`
slots = 8 MiB at 8 bytes each) — deliberately NOT a ring: slot
addressing is never circular, and each launch appends exactly its
operands' counts as one CONTIGUOUS slot range, so the kernel is handed
one device pointer plus a count, a plain array view. That linearity is
what makes the tile representation this simple — a circular table,
with indices wrapping modulo capacity, could not offer the
one-pointer-plus-count view for a range straddling the physical end.
There is **no per-tile chunk cap and no in-flight-handle cap**: the
free-slot count is the only bound. The pointers are written before the
launch: each launch's merged table is built in a pinned host mirror at
the cursor and copied by one `cudaMemcpyAsync` into the same offset of
the device table on a dedicated control stream, ordered ahead of the
kernel by an event, so the host takes no per-launch wait. Slots are
append-only — nothing overwrites a slot until a rewind, and a rewind
first waits for the event recorded after the latest table-reading
kernel — so every kernel that reads the old slots has finished before
slot 0 is reused. When a launch's pointers do not fit the slots
remaining to the physical end of the table, the transport waits on the
event recorded after the latest launch and **rewinds**: the cursor
restarts from slot 0 of the same table (its only backpressure) — a
restart from the top, not an end-to-start join. The one case the table
refuses is a single launch whose pointer set exceeds
`kChunkPtrTableSlots` even when empty: building that launch's table
aborts with a diagnostic (`LaunchTables::reserve_more`; a structural
limit), which
terminates the `noexcept` codelet body — no fallback driver is taken
for an oversized launch. A test hook (`NNHAUL_CHUNK_TABLE_SLOTS`)
shrinks the table to another power of two (other values warn once and
are ignored) so the rewind/reclaim path can be exercised on hardware
that cannot fill the default table.
(An earlier two-regime split — by-value kernel parameter tables under
a per-kernel chunk budget — was removed; the table is the
ONLY transport.)

The table **replaced an earlier host-mapped (`cudaHostAllocMapped`)
arena**, which was a correctness hazard: the kernel read the pinned
host array *at execution time*, and with the driver running ahead
on the in-order stream a queued-but-unexecuted kernel could read
slots the host had already repopulated for a later launch (a tile
offloaded and prefetched between two uses changes its chunk
addresses, so the earlier kernel would dereference the later use's
pointers). The table is device-resident, read-only, and written before the
launch. This transport is single-device: `ChunkView`/`LaunchTables`
refuse a launch on a second CUDA device rather than share the table.

## 5. Verification

What the suites actually prove:

- the kernel-level A/B (`nntile/tests/kernel/chunked_ab.cc`) runs `add`, `copy`
  and `transpose` through the classic and chunked entry points over ten
  chunks and a tail, and requires identical values;
- `nntile/tests/kernel/chunked_ops.cc` repeats that A/B over ten chunks
  plus a tail for 25 hot chunked entry points (elementwise, in-place,
  broadcast, reductions, softmax/maxsumexp, mask, logsumexp, SGD, indexed
  outputs, clear). The remaining chunked entries are exercised through the
  graph suites at `NNHAUL_CHUNK_SIZE_MB=1`, which runs the chunked codelet
  path and compares against the CPU references; not every entry has a
  dedicated bitwise kernel-level A/B yet. The conv2d family is disabled in
  the default transformers-only scope and `randn` has no independent value
  reference yet (a known coverage gap).
- the NNTile core, tile and tensor suites run on the NNHaul backend, and
  run again with `NNHAUL_CHUNK_SIZE_MB=1` so tensors above 1 MiB are
  multi-chunk (`tools/run_chunked_suite.sh`);
- `chunked_big` and the staged probes drive graph-level multi-chunk
  execution with explicit chunk-count assertions.

What they do not prove: a dedicated chunked kernel exists for every op
(62 of 69 op families have one, none go through chunk-aware drivers, 7 are
disabled, and none are unreferenced). The per-op coverage table is
`tools/chunked_coverage.py`. The conv2d chunked kernels additionally
cap the three operands' TOTAL chunk count at 4096 (64 GiB at the 16 MiB
class, 16 GiB at the default 4 MiB class); a larger convolution aborts
with that census rather than staging a fallback. Chunk size changes
performance, never the intended values.
