/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * chunk-graph plan's acts: the reference packing, the four
 * op factories, value equality, the tables' lookups, the duration
 * model's miss discipline, and the pure lane mapping over an op.
 *
 * @file src/plan.cc
 * The acts behind src/plan.hh's types: make_op_ref's
 * packing, make_link/make_unlink/make_copy/make_tile_op — each setting
 * exactly its kind's fields — operator== for PlanOp and EntryBinding,
 * tile_chunk_count, the tile and executability tables
 *, DurationModel's carries/d_ns/copy_ns (a miss throws
 * duration-model-miss, never serviced), and lane_of_op
 * . No planning, no execution, no device access.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#include "plan.hh"

#include "nnhaul/defs.h"

#include <algorithm>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nnhaul
{

// ---- the reference packing  ----

OpRef make_op_ref(std::uint64_t plan_seq, std::size_t position)
{
    return (plan_seq << 32) | static_cast<std::uint64_t>(position);
}

// ---- the four factories : exactly the kind's own fields ----

PlanOp make_link(TileId tile, ChunkIndex index, NodeId node, ChunkId chunk)
{
    PlanOp op;
    op.kind = PlanOpKind::Link;
    op.u.link = {tile, index, node, chunk};
    return op;
}

PlanOp make_unlink(
    TileId tile, ChunkIndex index, NodeId node, ChunkId chunk)
{
    PlanOp op;
    op.kind = PlanOpKind::Unlink;
    op.u.unlink = {tile, index, node, chunk};
    return op;
}

PlanOp make_copy(TileId tile, ChunkIndex index, ChunkLoc src, ChunkLoc dst)
{
    PlanOp op;
    op.kind = PlanOpKind::Copy;
    op.u.copy = {tile, index, src, dst};
    return op;
}

PlanOp make_tile_op(
    TaskId task, NodeId node, std::vector<Operand> operands)
{
    PlanOp op;
    op.kind = PlanOpKind::TileOp;
    op.u.tile_op = {task, node};
    op.operands = std::move(operands);
    return op;
}

// ---- value equality  ----

namespace
{

bool same(Operand const &a, Operand const &b)
{
    return a.tile == b.tile && a.index == b.index &&
        a.access == b.access && a.node == b.node && a.chunk == b.chunk;
}

} // namespace


bool operator==(PlanOp const &a, PlanOp const &b)
{
    if (a.kind != b.kind || a.operands.size() != b.operands.size())
    {
        return false;
    }
    // The union payload compares by the kind's own fields only — the
    // other arms' bytes are unspecified and never read.
    switch (a.kind)
    {
    case PlanOpKind::Link:
        if (a.u.link.tile != b.u.link.tile ||
            a.u.link.index != b.u.link.index ||
            a.u.link.node != b.u.link.node ||
            a.u.link.chunk != b.u.link.chunk)
        {
            return false;
        }
        break;
    case PlanOpKind::Unlink:
        if (a.u.unlink.tile != b.u.unlink.tile ||
            a.u.unlink.index != b.u.unlink.index ||
            a.u.unlink.node != b.u.unlink.node ||
            a.u.unlink.chunk != b.u.unlink.chunk)
        {
            return false;
        }
        break;
    case PlanOpKind::Copy:
        if (a.u.copy.tile != b.u.copy.tile ||
            a.u.copy.index != b.u.copy.index ||
            a.u.copy.src.node != b.u.copy.src.node ||
            a.u.copy.src.chunk != b.u.copy.src.chunk ||
            a.u.copy.dst.node != b.u.copy.dst.node ||
            a.u.copy.dst.chunk != b.u.copy.dst.chunk)
        {
            return false;
        }
        break;
    case PlanOpKind::TileOp:
        if (a.u.tile_op.task != b.u.tile_op.task ||
            a.u.tile_op.node != b.u.tile_op.node)
        {
            return false;
        }
        break;
    }
    for (std::size_t i = 0; i < a.operands.size(); ++i)
    {
        if (!same(a.operands[i], b.operands[i]))
        {
            return false;
        }
    }
    return true;
}

// ---- the tile decomposition  ----

std::size_t tile_chunk_count(std::size_t nbytes)
{
    // n = ceil(nbytes / chunk_class_bytes()). nbytes == 0 is the
    // degenerate VALID case: a zero-byte tile decomposes into n = 0
    // chunks — a tile with no chunks, carried by the tile table with
    // n = 0, never a one-chunk phantom (no index of it exists for an
    // op to name; index-out-of-range fires for every index).
    std::size_t const cls = chunk_class_bytes();
    if (nbytes > std::numeric_limits<std::size_t>::max() - (cls - 1))
    {
        throw std::runtime_error(
            "tile decomposition: nbytes overflows the chunk count");
    }
    return (nbytes + cls - 1) / cls;
}

std::size_t tile_chunk_span(std::size_t nbytes, ChunkIndex index)
{
    // min(chunk_class_bytes(), nbytes - index * chunk_class_bytes()) — the
    // occupied bytes of chunk i, the one arithmetic behind the
    // occupied-span definition (data.hh's chunk_nbytes and the trace's
    // copy byte stamp derive from THIS definition, never a second one).
    // An index at
    // or past the decomposition leaves no occupied bytes: the
    // subtraction guards its own underflow and the span is 0 — the
    // checker's index-out-of-range row is the first line of defense
    // and the channel's degenerate-span guard the belt; nobody clips.
    std::size_t const off =
        static_cast<std::size_t>(index) * chunk_class_bytes();
    std::size_t const left = nbytes > off ? nbytes - off : 0;
    return left < chunk_class_bytes() ? left : chunk_class_bytes();
}

// ---- the tile table  ----

void TileTable::add(TileId tile, std::size_t nbytes)
{
    if (rows_.find(tile) != rows_.end())
    {
        throw std::runtime_error(
            "tile table: duplicate tile " + std::to_string(tile));
    }
    std::size_t const count = tile_chunk_count(nbytes);
    if (count > std::numeric_limits<ChunkIndex>::max())
    {
        throw std::runtime_error(
            "tile table: tile " + std::to_string(tile) + " needs " +
            std::to_string(count) + " chunks, beyond the ChunkIndex range");
    }
    TileRow row;
    row.n = static_cast<ChunkIndex>(count);
    row.nbytes = nbytes;
    rows_.emplace(tile, row);
}

bool TileTable::carries(TileId tile) const
{
    return rows_.find(tile) != rows_.end();
}

ChunkIndex TileTable::n(TileId tile) const
{
    auto it = rows_.find(tile);
    if (it == rows_.end())
    {
        // Loud, never serviced: the unknown-tile row is the caller's
        // to check (carries() first); this is the belt.
        throw std::runtime_error(
            "unknown-tile: the tile table does not carry tile " +
            std::to_string(tile));
    }
    return it->second.n;
}

std::size_t TileTable::nbytes(TileId tile) const
{
    auto it = rows_.find(tile);
    if (it == rows_.end())
    {
        // As n(): the unknown-tile row is the caller's to check
        // (carries() first); this is the belt, never a serviced miss.
        throw std::runtime_error(
            "unknown-tile: the tile table does not carry tile " +
            std::to_string(tile));
    }
    return it->second.nbytes;
}

// ---- the executability table  ----

void ExecutabilityTable::add(TaskId task, std::vector<NodeId> nodes)
{
    if (runnable_.find(task) != runnable_.end())
    {
        throw std::runtime_error(
            "executability table: duplicate task " + std::to_string(task));
    }
    runnable_.emplace(task, std::move(nodes));
}

bool ExecutabilityTable::can(TaskId task, NodeId node) const
{
    auto it = runnable_.find(task);
    return it != runnable_.end() &&
        std::find(it->second.begin(), it->second.end(), node) !=
        it->second.end();
}

// ---- the duration model  ----

namespace
{

std::uint64_t task_node_key(TaskId t, NodeId n)
{
    // (TaskId << 8) | NodeId : NodeId is uint8_t, so the pair
    // packs collision-free.
    return (static_cast<std::uint64_t>(t) << 8) |
        static_cast<std::uint64_t>(n);
}

} // namespace

bool DurationModel::carries(TaskId t, NodeId n) const
{
    return task_node_ns.find(task_node_key(t, n)) != task_node_ns.end();
}

bool DurationModel::carries_channel(LaneId chan) const
{
    std::size_t const at = static_cast<std::size_t>(chan);
    return at < channel_bytes_per_ns.size() && channel_bytes_per_ns[at] != 0;
}

std::uint64_t DurationModel::d_ns(TaskId t, NodeId n) const
{
    auto it = task_node_ns.find(task_node_key(t, n));
    if (it == task_node_ns.end())
    {
        // Simulate mode's census, never serviced ; the
        // timing pass checks carries() first — this is the belt.
        throw std::runtime_error("duration-model-miss");
    }
    return it->second;
}

std::uint64_t DurationModel::copy_ns(LaneId chan) const
{
    if (!carries_channel(chan))
    {
        // As d_ns: a channel lane the model does not carry (out of
        // range, or a zero rate — no entry) is duration-model-miss.
        throw std::runtime_error("duration-model-miss");
    }
    return static_cast<std::uint64_t>(chunk_class_bytes()) /
        channel_bytes_per_ns[static_cast<std::size_t>(chan)];
}

// ---- the lane mapping  ----

LaneId lane_of_op(PlanOp const &op, NodeInventory const &inv)
{
    switch (op.kind)
    {
    case PlanOpKind::Link:
    case PlanOpKind::Unlink:
        // No lane: host-side binding-table transitions at their issue
        // instant — no device work, no lane, no event .
        return kNoLane;
    case PlanOpKind::TileOp:
        return inv.lane_of_exec(op.u.tile_op.node);
    case PlanOpKind::Copy:
        return inv.lane_of_chan(op.u.copy.src.node, op.u.copy.dst.node);
    }
    return kNoLane;
}

} // namespace nnhaul
