/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nnhaul/planner.hh
 * Public planner interface: the inputs a lowering planner reads, the plan
 * operations it emits, and the entry point a planner plugin exports.
 *
 * This header is the contract. A planner written against it needs no other
 * NNHaul header and no private source tree. The in-tree LRU planner emits
 * its plan through the same Sink, and external planners (including
 * proprietary ones loaded with NNHAUL_PLANNER=<path>) use this header
 * unchanged.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include <nnhaul/nnhaul.hh>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nnhaul
{

// ---- atoms of the plan vocabulary (public) ----

//! A chunk's identity is (node, ChunkId); ChunkId is scoped to its node.
using ChunkId = std::uint32_t;
//! One tile; ChunkIndex is a chunk's 0-based position within the tile.
using TileId = std::uint32_t;
using ChunkIndex = std::uint32_t;
//! The plan-time task symbol.
using TaskId = std::uint32_t;

//! One physical chunk: node and chunk, always both named.
struct ChunkLoc
{
    NodeId node = 0;
    ChunkId chunk = 0;
};

//! The access mode an operand is used with (public nnhaul::Access).
struct TileUse
{
    TileId tile = 0;
    Access access = Access::Read;
};

//! One task of the batch, in submit order, with its execution node.
struct PlannedTask
{
    TaskId task = 0;
    NodeId node = 0;
    std::vector<TileUse> operands;
};

//! One operand of a TILE_OP: the logical chunk, its access mode and the
//! binding at this queue position.
struct Operand
{
    TileId tile = 0;
    ChunkIndex index = 0;
    Access access = Access::Read;
    NodeId node = 0;
    ChunkId chunk = 0;
};

//! One residency fact: a node's chunk holds a tile's logical chunk, with
//! the one actual/outdated flag.
struct EntryBinding
{
    NodeId node = 0;
    ChunkId chunk = 0;
    TileId tile = 0;
    ChunkIndex index = 0;
    bool actual = false;
};

inline bool operator==(EntryBinding const &a, EntryBinding const &b) noexcept
{
    return a.node == b.node && a.chunk == b.chunk && a.tile == b.tile
        && a.index == b.index && a.actual == b.actual;
}

//! The batch a planner plans: the sequence number, the entry residency
//! fold, the tasks in submit order, and the tiles dying in this batch.
struct PlanRequest
{
    std::uint64_t plan_seq = 0;
    std::vector<EntryBinding> entry; // sorted by (node, chunk)
    std::vector<PlannedTask> tasks;
    std::vector<TileId> dead_tiles;
};

//! The planner's measurement counters, returned beside the plan.
struct PlanStats
{
    std::uint64_t evictions = 0;
    std::uint64_t rescues = 0;
    std::uint64_t refetches = 0;
    std::uint64_t prefetches = 0;
    std::uint64_t dying_frees = 0;
    std::uint64_t entry_bindings = 0;
    std::uint64_t exit_bindings = 0;
    std::uint64_t run_misses = 0;
    std::uint32_t run_miss_want = 0;
    std::size_t run_miss_longest = 0;
    std::uint64_t pin_spends = 0;
};

namespace planner
{

//! Interface version. A plugin returns the version it was built against
//! from nnhaul_planner_api_version(); a mismatch is refused at load.
inline constexpr std::uint32_t kApiVersion = 1;

//! Read-only view of the world a plan is built over: the node inventory
//! (capacities, channel connectivity and rates) and the tile
//! decomposition (chunk count and payload bytes per tile).
class Topology
{
  public:
    virtual ~Topology() = default;

    virtual std::size_t node_count() const = 0;
    virtual bool is_host(NodeId node) const = 0;
    //! Soft ceiling of the node's chunk pool, in chunks (0 = unlimited).
    virtual std::size_t pool_capacity(NodeId node) const = 0;
    //! Chunks the node's pool holds NOW. Chunk ids `0..pool_chunks-1`
    //! are the only valid ones until grow() raises the cardinality.
    virtual std::size_t pool_chunks(NodeId node) const = 0;
    //! True when (node, chunk) names a chunk the pool owns now.
    virtual bool chunk_valid(NodeId node, ChunkId chunk) const = 0;

    virtual bool has_channel(NodeId src, NodeId dst) const = 0;
    //! Relative transfer rate on the channel when one is known, 1.0 when
    //! the channel exists but carries no registered rate, and 0.0 when
    //! no channel exists.
    virtual double channel_rate(NodeId src, NodeId dst) const = 0;

    virtual bool carries(TileId tile) const = 0;
    //! These two THROW for a tile the run does not carry; guard with
    //! carries() first.
    virtual ChunkIndex tile_chunks(TileId tile) const = 0;
    virtual std::size_t tile_bytes(TileId tile) const = 0;
};

//! Destination for a planner's output. The engine validates whatever a
//! planner emits with the same checker the in-tree planner passes.
class Sink
{
  public:
    virtual ~Sink() = default;

    virtual void link(TileId tile, ChunkIndex index, NodeId node,
        ChunkId chunk) = 0;
    virtual void unlink(TileId tile, ChunkIndex index, NodeId node,
        ChunkId chunk) = 0;
    //! Both endpoints must ALREADY be bound to (tile, index): LINK the
    //! destination first, with actual=false; a copy into an actual
    //! binding, or from an outdated one, fails the checker.
    virtual void copy(TileId tile, ChunkIndex index, ChunkLoc src,
        ChunkLoc dst) = 0;
    //! One launch for one task occurrence, with its full operand set.
    //! `node` must be the node the task was submitted on. `operands` must
    //! follow the submission's buffer order, and within each buffer the
    //! tile's chunk indices ascending, with exactly one entry per
    //! (buffer, chunk) pair: the executor hands each run of `n` pointers
    //! straight to the codelet for that buffer. A task whose buffers name
    //! the same tile twice is refused at the next close-and-plan, so
    //! every buffer tile is
    //! distinct. The checker accepts a split over several TILE_OPs; the
    //! engine's pre-issue check refuses it, because each launch would
    //! carry a short array. Emit one TILE_OP per occurrence.
    virtual void tile_op(TaskId task, NodeId node,
        std::vector<Operand> const &operands) = 0;

    //! Ask the engine to grow node's chunk pool to `cardinality` chunks
    //! (compilation owns growth; the plan records the final size). The
    //! new cardinality must exceed the current one and respect the soft
    //! limit; the new chunks are free by construction.
    virtual void grow(NodeId node, std::size_t cardinality) = 0;

    //! The predicted exit residency. Required and shape-checked; the
    //! engine seeds the next batch from the checker's exit fold, not
    //! from this value.
    virtual void set_exit(std::vector<EntryBinding> const &exit) = 0;
    //! Optional counters; a plugin may leave them at zero.
    virtual void set_stats(PlanStats const &stats) = 0;
};

//! A lowering planner: read the request and the topology, emit operations
//! into the sink. Must be deterministic for the same inputs and must be
//! thread-safe: one loaded planner instance can serve more than one
//! Runtime (the engine serializes within one Runtime, not across them).
class Planner
{
  public:
    virtual ~Planner() = default;
    virtual char const *name() const = 0;
    virtual void plan(PlanRequest const &request, Topology const &topology,
        Sink &sink) = 0;
};

//! Load a planner shared library. Throws nnhaul::Error on a missing
//! library or symbol, an API-version mismatch, or a null planner, naming
//! the expected and found versions. Used by NNHAUL_PLANNER and by tests.
Planner *load(char const *path);

} // namespace planner
} // namespace nnhaul

// ---- the plugin entry point ----
//
// A planner shared library exports exactly these two symbols:
//
//   extern "C" std::uint32_t nnhaul_planner_api_version();
//   extern "C" nnhaul::planner::Planner *nnhaul_create_planner();
//
// The created object is owned by the library (a process-lifetime singleton
// is the expected shape); NNHaul never destroys it.

extern "C"
{
using nnhaul_planner_api_version_fn = std::uint32_t (*)();
using nnhaul_planner_create_fn = nnhaul::planner::Planner *(*)();
}
