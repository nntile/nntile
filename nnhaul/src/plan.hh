/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * chunk-graph plan as C++ types: the atoms, the four
 * operations, the two-part plan, the census, the duration model, and
 * the checker's input tables.
 *
 * @file src/plan.hh
 * The plan types:
 * the ids and the run-scoped op reference, PlanOp (exactly the four
 * kinds Link, Unlink, Copy, TileOp), EntryBinding and the
 * two-part Plan, SegmentSnapshot and Census,
 * DurationModel, the checker inputs (TileTable, ExecutabilityTable,
 * the task sequence), and lane_of_op — the pure mapping from an op to
 * its lane. Minimality is deliberate: no op field and no plan field
 * names an event, a wait,
 * a record, an edge, a stream, a lane, a duration, a priority, or a
 * hint — nothing but the op's own facts. Scheduling lives in neither
 * file; every dependency and sync is derived.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

// The Access atom lives in the PUBLIC header (nntile passes
// nnhaul::Access values in Buffer use declarations): this file
// includes it and never redefines it — exactly ONE nnhaul::Access in
// the build, so any TU may include both headers .
#include "inventory.hh"
#include "nnhaul/nnhaul.hh"
#include <nnhaul/planner.hh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace nnhaul
{

// ---- the atoms  ----

// A chunk's identity is (NodeId, ChunkId) — ChunkId is
// scoped to its node and two nodes' ChunkId spaces are unrelated.
// NodeId itself comes from inventory.hh (devices 0..k-1 in
// ascending CUDA ordinal, host k) and is not redefined here.
// TileId names one tile; ChunkIndex is a chunk's
// 0-based position within its tile's decomposition — the pair is one
// logical chunk, the identity of DATA independent of residency.
using TileId = std::uint32_t;
using ChunkIndex = std::uint32_t;

// The run-scoped op reference: (plan_seq << 32) | queue position.
// Facts never reset across plans , so the reference IS
// run-scoped and integer-ordered by issue. Defined in plan.cc .
using OpRef = std::uint64_t;
inline constexpr OpRef kNoRef =
    std::numeric_limits<std::uint64_t>::max();
OpRef make_op_ref(std::uint64_t plan_seq, std::size_t position);

// Access is one of Read, Write, ReadWrite — the public
// nnhaul::Access, included above. Operand and TileAccess carry it; no
// second definition exists anywhere in the tree.

// One logical chunk — what the data IS, wherever it resides.
struct LogicalChunk
{
    TileId tile;
    ChunkIndex index;
};

// ---- PlanOp: one of exactly the four operations  ----


// The queue contains EXACTLY these four kinds; any other entry
// is the ERROR malformed-plan.
enum class PlanOpKind : std::uint8_t { Link, Unlink, Copy, TileOp };

// A kind tag plus a union of the four POD payloads, with TILE_OP's
// operand list beside the union: every payload is a handful of
// integers, dispatch is a switch in the one fold, and the tagged
// union keeps the payload trivially copyable, so the census, the
// issue log, and the prefix snapshot move them as bytes. Each op
// carries exactly the fields its kind defines — no waits,
// no records, no lanes, no hints .
struct PlanOp
{
    PlanOpKind kind = PlanOpKind::Link;
    union
    {
        struct
        {
            TileId tile;      // the logical chunk's tile
            ChunkIndex index; // its position within the tile
            NodeId node;      // the node of the chunk being linked
            ChunkId chunk;    // the chunk, on node
        } link;
        struct
        {
            TileId tile;      // the logical chunk's tile
            ChunkIndex index; // its position within the tile
            NodeId node;      // the node of the chunk being unlinked
            ChunkId chunk;    // the chunk, on node
        } unlink;
        struct
        {
            TileId tile;      // the logical chunk both endpoints name
            ChunkIndex index; //
            ChunkLoc src;     // the source binding
            ChunkLoc dst;     // the destination binding
        } copy;
        struct
        {
            TaskId task; // the task to execute
            NodeId node; // the execution node: the op runs on its lane
        } tile_op;
    } u{};
    std::vector<Operand> operands; // TILE_OP's third field; empty for
                                   // the others
};

static_assert(std::is_trivially_copyable<decltype(PlanOp::u)>::value,
    "the PlanOp payload union must stay trivially copyable ");

// Value equality :
// same kind, same union payload for that kind, and same operands
// entry for entry. Declared here, defined in plan.cc.
bool operator==(PlanOp const &a, PlanOp const &b);

// The four factories (plan.cc): each builds an op of its kind setting
// exactly that kind's fields and nothing else — the payloads of the
// other kinds never exist, and operands stay empty except for
// TILE_OP's own.
PlanOp make_link(TileId tile, ChunkIndex index, NodeId node, ChunkId chunk);
PlanOp make_unlink(
    TileId tile, ChunkIndex index, NodeId node, ChunkId chunk);
PlanOp make_copy(TileId tile, ChunkIndex index, ChunkLoc src, ChunkLoc dst);
PlanOp make_tile_op(
    TaskId task, NodeId node, std::vector<Operand> operands);

// ---- the Plan: entry state plus one queue  ----

// One linked binding with its flag  — the
// plan's assumed world, sparse per node: free chunks are the pool's
// complement and carry no entry.
// Value equality is inline in <nnhaul/planner.hh>.

// The plan IS exactly two parts, nothing beside : the
// entry state — the linked bindings with their flags — and the queue,
// a totally ordered list of ops whose order IS the whole dependency
// input. Minimality is normative : no waits, records,
// events, lanes, durations, or hints exist as fields.
struct Plan
{
    std::uint64_t plan_seq = 0;      // the run-scoped sequence
    std::vector<EntryBinding> entry; // sorted by (node, chunk); free
                                     // chunks absent
    std::vector<PlanOp> queue;       // the whole dependency input
    // The pools this plan GREW, per node, as the walk's
    // final cardinalities (entries beyond the plan's entry state —
    // new chunks are FREE by construction; asks succeed only BEFORE
    // the pool's growth reaches its limit). The ENGINE commits
    // before check(), ahead of any op issue — apply_plan_growth runs
    // the single growth step (up to the limit) and syncs the inventory,
    // binding table, dependency rows, and event pools —
    // compilation owns growth, never the run phase; the checker
    // applies the same extension before its fold, and the executor's
    // boundary check verifies that the commit happened. Empty = a plan
    // that grew nothing.
    std::vector<std::size_t> growth;
};

// ---- the checker's inputs  ----

// A tile of nbytes decomposes into
// n = ceil(nbytes / chunk_class_bytes()); the last chunk's remainder is
// the intra-chunk tail, wasted and never repurposed. nbytes and n are
// tile-graph metadata the plan never carries. Defined in plan.cc;
// chunk_class_bytes() is the process-wide runtime class (env-initialized
// on first use, <nnhaul/defs.h>).
std::size_t tile_chunk_count(std::size_t nbytes);

// The one definition: the OCCUPIED SPAN of physical
// chunk `index` of a tile of nbytes — min(chunk_class_bytes(), nbytes -
// index * chunk_class_bytes()). A full chunk occupies the whole class and
// a COPY of it moves the whole class; the tail chunk occupies exactly
// its payload remainder and a COPY of it moves that span and nothing
// more — never the wasted bytes of the block behind it. An index at or
// past the decomposition spans 0 (no occupied bytes exist there);
// every caller guards the degenerate span loudly — the span the
// executor computes is a submission fact the channel refuses to clip.
// Defined in plan.cc.
std::size_t tile_chunk_span(std::size_t nbytes, ChunkIndex index);

// The tile table — the registry fact TileId -> chunk count
// n with the payload bytes n derives from, a checker input beside the
// plan, data only. The payload bytes are the registry's own fact since
// the byte-exact copy rule: a COPY's occupied span is
// tile_chunk_span(nbytes(tile), index) , so the registered size
// must survive beside the count. Sparse: a tile the table does not
// carry is the fault unknown-tile, the caller's row to check first.
class TileTable
{
  public:
    // Registers one tile by its byte size (n = tile_chunk_count; the
    // bytes kept verbatim for the span rule). Loud on a duplicate
    // registration — a registry defect.
    void add(TileId tile, std::size_t nbytes);

    bool carries(TileId tile) const;

    // Aborts loudly on a tile the table does not carry: the checker's
    // unknown-tile row asks carries() first, so this is the belt,
    // never a serviced miss.
    ChunkIndex n(TileId tile) const;

    // The registered payload bytes — the fact a COPY's occupied span
    // derives from . Aborts loudly on a
    // tile the table does not carry, exactly as n().
    std::size_t nbytes(TileId tile) const;

  private:
    // One row: the decomposition and the size it derives from — both
    // facts of the one registration, never stored apart.
    struct TileRow
    {
        ChunkIndex n = 0;
        std::size_t nbytes = 0;
    };

    std::unordered_map<TileId, TileRow> rows_;
};

// The executability table — the registry fact
// task -> runnable nodes, a checker input beside the plan. A task IS
// runnable on a CUDA device node iff its codelet has a CUDA body, on
// the host node iff a CPU body; the TABLE is just data — its builder
// is the registration piece , never this file.
class ExecutabilityTable
{
  public:
    // Registers one task's runnable-node set. Loud on a duplicate —
    // a registry defect.
    void add(TaskId task, std::vector<NodeId> nodes);

    bool can(TaskId task, NodeId node) const;

  private:
    std::unordered_map<TaskId, std::vector<NodeId>> runnable_;
};

// The task sequence — the batch's tasks in the tile
// graph's topological order, each with its operand tiles and their
// accesses.
struct TileAccess
{
    TileId tile;
    Access access;
};

struct SeqTask
{
    TaskId task;
    std::vector<TileAccess> operands;
};

// ---- the census  ----

// The census's per-node render: the binding table segment with the
// free list printed beside it. Bindings are dense by ChunkId; free
// slots are marked by the renderer that fills the snapshot.
struct SegmentSnapshot
{
    NodeId node = 0;
    std::size_t pool = 0;
    std::vector<EntryBinding> bindings; // dense by ChunkId; free slots
                                        // marked
    std::vector<ChunkId> free_list;     // printed beside the table
};

// The loud-failure artifact: a fault is never serviced, never
// retried, never repaired — the run stops. It carries the finding
// (the fault's name; the position — queue index, entry, or the need;
// the offending op verbatim), the fold snapshot per node, the entry
// state, the ops prefix, the inputs' relevant facts, and at execution
// time additionally the applied binding table and the issue log
// prefix. The census travels beside the plan, never in it.
struct Census
{
    std::string fault;             // name + position; the op verbatim
    std::optional<PlanOp> op;
    std::vector<SegmentSnapshot> fold;    // the fold snapshot, every
                                          // node
    std::vector<SegmentSnapshot> applied; // execution time only
    std::vector<EntryBinding> entry;
    std::vector<PlanOp> prefix;      // the ops prefix
    std::vector<PlanOp> issue_prefix; // issue: execution only
    std::string inputs; // the inputs' relevant facts, rendered
};

// ---- the duration model  ----

// Run-scoped engine configuration beside the node inventory: static
// for the run, read-only, INPUT only — never an observation. The
// table: D(T, N) per (task, node) pair; R(A,B) per channel, per
// ordered pair, one COPY = chunk_class_bytes() / R(A,B); LINK and UNLINK
// are exactly zero, compile-time — no entry, no lookup. A lookup that
// misses is duration-model-miss (simulate's census, never
// serviced); the timing pass checks carries()/carries_channel() first
// and the throwing lookup is the belt. Check and execute never read
// the model.
struct DurationModel
{
    std::unordered_map<std::uint64_t, std::uint64_t> task_node_ns;
    // (TaskId<<8)|NodeId -> D(T, N) in ns

    std::vector<std::uint64_t> channel_bytes_per_ns;
    // R(A,B) per channel lane, dense by LaneId: index by the lane id
    // itself; a slot of zero is not an entry (no copy is modeled over
    // a zero-rate channel — the division would be meaningless).

    bool carries(TaskId t, NodeId n) const;
    bool carries_channel(LaneId chan) const;

    std::uint64_t d_ns(TaskId t, NodeId n) const; // miss -> the throw
    std::uint64_t copy_ns(LaneId chan) const;     // chunk_class_bytes() /
                                                  // R(A,B)
};

// ---- the lane mapping  ----

// Lane assignment IS a pure function of the op's own fields —
// derived, never planned, zero freedom: TILE_OP -> exec(node); COPY ->
// chan(src.node, dst.node); LINK and UNLINK -> no lane (kNoLane) —
// host-side binding-table transitions at their issue instant, no
// device work, no lane, no event. Lives here because it needs PlanOp;
// NodeInventory's side (lane_of_exec, lane_of_chan) is inventory.hh's.
// Defined in plan.cc.
LaneId lane_of_op(PlanOp const &op, NodeInventory const &inv);

} // namespace nnhaul
