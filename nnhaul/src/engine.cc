/*! @copyright (c) 2026-present Skolkovo Institute of Science and
 *                              Technology (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine over N memory nodes:
 * a virtual planner builds one plan per batch from a drained snapshot,
 * and a literal executor issues exactly the planned operations on the
 * lanes of the node inventory.
 *
 * @file src/engine.cc
 * The engine's acts over the landed pieces: the nine-step init
 * (select+probe, contexts, peer-enable, arenas, host pool, lanes,
 * event pools, per-node workspaces/handles/registration, the
 * binding table's all-free segments) and the barrier-then-join
 * shutdown; the close-and-plan SPLIT into its two acts (ISSUE at
 * signal_graph_end — drain, intern the task symbols, plan, check with
 * the FULL inputs and the chain
 * ALWAYS, enqueue everything, return; JOIN at
 * join_predecessor/wait — the record's queue-close barriers plus the
 * retirement accounting) with the dead-handle retirement riding the
 * batch's ops as trailing UNLINKs over the planner's predicted exit
 * and the loud third-batch depth abort; the host acquires
 * re-realized over host-node TILE_OPs on a contiguous run of
 * whole-class host-pool blocks (free-or-evictable, the eviction a plan
 * act with the data-survival rule); the per-node in-codelet TLS; and
 * the simulate exposure at the join over the batch's own folded owed
 * sets.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "engine.hh"

#include "planner_api.hh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cstdio>
#include <set>
#include <utility>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace nnhaul
{

thread_local Tls tls;

std::unique_ptr<RuntimeState> rt;
std::mutex api_mu;

// ---- the dead-pending intrusive list (the O(1) handle list) ----

void RuntimeState::dead_pending_push(HandleImpl *impl)
{
    impl->dead_prev = dead_pending_tail;
    impl->dead_next = nullptr;
    if (dead_pending_tail != nullptr)
    {
        dead_pending_tail->dead_next = impl;
    }
    else
    {
        dead_pending_head = impl;
    }
    dead_pending_tail = impl;
    ++dead_pending_count;
}

void RuntimeState::dead_pending_unlink(HandleImpl *impl)
{
    if (impl->dead_prev == nullptr && dead_pending_head != impl)
    {
        return; // not listed
    }
    if (impl->dead_prev != nullptr)
    {
        impl->dead_prev->dead_next = impl->dead_next;
    }
    else
    {
        dead_pending_head = impl->dead_next;
    }
    if (impl->dead_next != nullptr)
    {
        impl->dead_next->dead_prev = impl->dead_prev;
    }
    else
    {
        dead_pending_tail = impl->dead_prev;
    }
    impl->dead_prev = nullptr;
    impl->dead_next = nullptr;
    --dead_pending_count;
}


// ---- the env surfaces carried over (the env-robustness test's pins) ----

std::size_t limit_cuda_mem_bytes()
{
    // Read per-init: tests set and unset the env around scenarios.
    char const *e = std::getenv("NNHAUL_LIMIT_CUDA_MEM");
    if (e == nullptr)
    {
        return std::size_t{0};
    }
    char *end = nullptr;
    errno = 0;
    long mib = std::strtol(e, &end, 10);
    if (end == e || *end != '\0' || errno == ERANGE || mib <= 0)
    {
        throw Error(
            "NNHAUL_LIMIT_CUDA_MEM must be a positive number of MiB");
    }
    // A long of MiB can exceed size_t once shifted; reject the absurd
    // rather than wrap the cap.
    if (static_cast<unsigned long>(mib) >
        (std::numeric_limits<std::size_t>::max() >> 20))
    {
        throw Error("NNHAUL_LIMIT_CUDA_MEM is impossibly large");
    }
    return static_cast<std::size_t>(mib) << 20;
}

// NO preallocated memory
// by default: pools start at ZERO chunks. NNHAUL_PREALLOC_CPU_MEM and
// NNHAUL_PREALLOC_CUDA_MEM (MiB) name each side's explicit head start
// (that pool preallocates at most this many MiB of whole class chunks,
// clamped by its own soft limit), and growth BEYOND it continues on
// demand to the soft/hard limits — a preallocation is a head start,
// never a cap.
std::size_t prealloc_mib(char const *knob_name)
{
    char const *e = std::getenv(knob_name);
    if (e == nullptr)
    {
        return std::size_t{0};
    }
    long mib = 0;
    if (!env_parse_long(e, mib) || mib < 0)
    {
        std::fprintf(stderr,
            "[nnhaul] %s=%s is not a non-negative integer; "
            "preallocating nothing\n",
            knob_name, e);
        return std::size_t{0};
    }
    if (static_cast<unsigned long>(mib)
        > std::numeric_limits<std::size_t>::max() >> 20)
    {
        throw Error(std::string("nnhaul: ") + knob_name + "=" + e
            + " is too large to convert to bytes");
    }
    return static_cast<std::size_t>(mib);
}

std::size_t cublas_workspace_bytes()
{
    static std::size_t const v = [] {
        char const *e = std::getenv("NNHAUL_CUBLAS_WORKSPACE_MB");
        if (e == nullptr)
        {
            return std::size_t{8} << 20;
        }
        long parsed = 0;
        if (!env_parse_long(e, parsed))
        {
            std::fprintf(stderr,
                "[nnhaul] NNHAUL_CUBLAS_WORKSPACE_MB=%s is not an "
                "integer; using 8 MiB\n", e);
            return std::size_t{8} << 20;
        }
        // Clamp both ends instead of letting the MiB-to-bytes shift
        // wrap (negative stays "no workspace", as before).
        constexpr long kMaxMiB = 4'096; // 4 GiB
        long mib = parsed < 0      ? 0
                   : parsed > kMaxMiB ? kMaxMiB
                                      : parsed;
        return static_cast<std::size_t>(mib) << 20;
    }();
    return v;
}

void warn_retired_knobs()
{
    static bool const warned = [] {
        for (char const *name : {"NNHAUL_CHUNK_SIZES",
                 "NNHAUL_CHUNK_POLICY", "NNHAUL_CHUNK_MIN_COUNT"})
        {
            if (std::getenv(name) != nullptr)
            {
                std::fprintf(stderr,
                    "[nnhaul] %s is retired: the chunk class is the "
                    "runtime %zu-byte constant; the knob is "
                    "ignored\n",
                    name,
                    chunk_class_bytes());
            }
        }
        // NNHAUL_CHUNK_TILES is deliberately NOT in the retired list: it
        // was never the engine's knob. It belongs to nntile's chunked-
        // TILE allocation gate (see nntile's nnhaul/chunk_gate.hh) and
        // is honored there (=0 = unchunked tile handles for the
        // process). The engine's chunk class is and always was a
        // compile-time constant, independent of that flag.
        for (char const *name : {"NNHAUL_PLAN_CHECK",
                 "NNHAUL_BATCH_FENCE", "NNHAUL_OFFLOAD_POLICY",
                 "NNHAUL_LAUNCH_AHEAD"})
        {
            if (std::getenv(name) != nullptr)
            {
                std::fprintf(stderr,
                    "[nnhaul] %s is retired: one check, always full, "
                    "no fence exists beside the one registry, and the "
                    "offloader and the launch-ahead pipeline are "
                    "deleted machinery; the knob is ignored\n",
                    name);
            }
        }
        return true;
    }();
    (void) warned;
}

namespace
{

// ---- the seed's index: the predicted world as queries AND as the
//      acquire's working copy (the state read; the
//      acquire's boundary acts mutate it exactly as the ops they
//      append fold the world) ----

struct SeedIndex
{
    // Per node: chunk -> (tile, index, actual). NodeId's whole space
    // stays below the vector's size; pools are run-scoped thousands of
    // chunks, and the acquire's queries are not the hot path.
    std::vector<std::unordered_map<ChunkId,
        std::tuple<TileId, ChunkIndex, bool>>>
        by_chunk_;

    // The (node, tile, index) -> (chunk, actual) row: the seed
    // holds at most one binding per (node, tile, index) (the at-most-
    // one-binding property), so this key is unique per row and binding_of
    // becomes one hash probe — the old per-chunk linear scan of the
    // node's row map was verify_pins' whole cost at the
    // giant single-shot close, a small cost at every per-iteration close).
    // Same answers by construction (the row IS the same set of
    // bindings); by_chunk_ stays the acquire's working copy.
    struct TileKey
    {
        NodeId node;
        TileId tile;
        ChunkIndex index;
        bool operator==(TileKey const &o) const
        {
            return node == o.node && tile == o.tile
                && index == o.index;
        }
    };
    struct TileKeyHash
    {
        std::size_t operator()(TileKey const &k) const
        {
            std::size_t h = 1469598103934665603ULL;
            auto mix = [&h](std::uint64_t v) {
                for (unsigned i = 0; i < 8; ++i)
                {
                    h ^= (v >> (i * 8)) & 0xff;
                    h *= 1099511628211ULL;
                }
            };
            mix(k.node);
            mix(k.tile);
            mix(k.index);
            return h;
        }
    };
    std::unordered_map<TileKey, std::pair<ChunkId, bool>, TileKeyHash>
        by_tile_;

    explicit SeedIndex(std::vector<EntryBinding> const &seed) :
        by_chunk_(257)
    {
        for (EntryBinding const &b : seed)
        {
            by_chunk_[b.node][b.chunk] =
                std::make_tuple(b.tile, b.index, b.actual);
            by_tile_[TileKey{b.node, b.tile, b.index}] =
                {b.chunk, b.actual};
        }
    }

    std::optional<std::pair<ChunkId, bool>> binding_of(
        TileId tile, ChunkIndex index, NodeId node) const
    {
        auto it = by_tile_.find(TileKey{node, tile, index});
        if (it == by_tile_.end())
        {
            return std::nullopt;
        }
        return it->second;
    }

    // The lowest DEVICE node holding an ACTUAL replica (the host
    // channels always exist; devices precede the host in
    // NodeId order).
    std::optional<ChunkLoc> device_actual_replica(
        TileId tile, ChunkIndex index, std::size_t device_count) const
    {
        for (std::size_t d = 0; d < device_count; ++d)
        {
            NodeId const node = static_cast<NodeId>(d);
            auto b = binding_of(tile, index, node);
            if (b.has_value() && b->second)
            {
                return ChunkLoc{node, b->first};
            }
        }
        return std::nullopt;
    }

    // The lowest DEVICE node holding ANY binding of the chunk (actual
    // or outdated) — the publish restore's target: a standing replica
    // is worth restoring whatever its flag (the publish outdates them
    // all anyway).
    std::optional<ChunkLoc> device_replica(
        TileId tile, ChunkIndex index, std::size_t device_count) const
    {
        for (std::size_t d = 0; d < device_count; ++d)
        {
            NodeId const node = static_cast<NodeId>(d);
            auto b = binding_of(tile, index, node);
            if (b.has_value())
            {
                return ChunkLoc{node, b->first};
            }
        }
        return std::nullopt;
    }

    // True iff some node OTHER than `skip` holds an ACTUAL replica.
    bool actual_replica_elsewhere(
        TileId tile, ChunkIndex index, NodeId skip) const
    {
        for (std::size_t n = 0; n < by_chunk_.size(); ++n)
        {
            NodeId const node = static_cast<NodeId>(n);
            if (node == skip)
            {
                continue;
            }
            auto b = binding_of(tile, index, node);
            if (b.has_value() && b->second)
            {
                return true;
            }
        }
        return false;
    }

    // The lowest free chunk on a device node (staging for a rescue or
    // a round trip).
    std::optional<std::pair<NodeId, ChunkId>> free_device_chunk(
        NodeInventory const &inv) const
    {
        for (std::size_t d = 0; d < inv.device_count(); ++d)
        {
            NodeId const node = static_cast<NodeId>(d);
            for (std::size_t c = 0; c < inv.pool_cardinality(node); ++c)
            {
                if (by_chunk_[node].find(static_cast<ChunkId>(c)) ==
                    by_chunk_[node].end())
                {
                    return std::make_pair(node, static_cast<ChunkId>(c));
                }
            }
        }
        return std::nullopt;
    }

    // BOTH indexes move together — the acquire's landing ladder reads
    // binding_of/actual_replica_elsewhere (by_tile_) WHILE it mutates
    // the world through these acts; a by_chunk_-only mutation leaves
    // by_tile_ claiming a binding that was just unlinked (the exact
    // drift that let a rescue's replica check see a dead host copy
    // and evict the fresh device rescue of the same chunk).
    void unlink_binding(NodeId node, ChunkId chunk)
    {
        auto it = by_chunk_[node].find(chunk);
        if (it == by_chunk_[node].end())
        {
            return;
        }
        by_tile_.erase(TileKey{node, std::get<0>(it->second),
            std::get<1>(it->second)});
        by_chunk_[node].erase(it);
    }

    void link_binding(TileId tile, ChunkIndex index, NodeId node,
        ChunkId chunk, bool actual)
    {
        // One binding per (node, tile, index): a re-link displaces the
        // old chunk's row so the indexes never disagree.
        auto it = by_tile_.find(TileKey{node, tile, index});
        if (it != by_tile_.end() && it->second.first != chunk)
        {
            by_chunk_[node].erase(it->second.first);
        }
        by_chunk_[node][chunk] = std::make_tuple(tile, index, actual);
        by_tile_[TileKey{node, tile, index}] = {chunk, actual};
    }

    // The node's chunk -> binding map (the landing ladder's by-position
    // walk).
    std::unordered_map<ChunkId,
        std::tuple<TileId, ChunkIndex, bool>> const &
    by_chunk_at(NodeId node) const
    {
        return by_chunk_[node];
    }

    // The world as a sorted entry-state vector (nodes ascending, chunks
    // ascending) — the census's fold render for the acquire's plan-time
    // pool-exhaustion artifact .
    std::vector<EntryBinding> bindings() const
    {
        std::vector<EntryBinding> out;
        for (std::size_t n = 0; n < by_chunk_.size(); ++n)
        {
            std::vector<ChunkId> chunks;
            chunks.reserve(by_chunk_[n].size());
            for (auto const &kv : by_chunk_[n])
            {
                chunks.push_back(kv.first);
            }
            std::sort(chunks.begin(), chunks.end());
            for (ChunkId c : chunks)
            {
                auto const &held = by_chunk_[n].at(c);
                EntryBinding b;
                b.node = static_cast<NodeId>(n);
                b.chunk = c;
                b.tile = std::get<0>(held);
                b.index = std::get<1>(held);
                b.actual = std::get<2>(held);
                out.push_back(b);
            }
        }
        return out;
    }
};

// The host pool's occupancy in the seed: per block, FREE or the
// binding it holds. The acquire's working copy mutates alongside the
// ops it appends .
struct HostBlocks
{
    std::vector<char> free_;                       // per host block
    std::vector<std::optional<EntryBinding>> held_; // per host block

    explicit HostBlocks(RuntimeState const &R)
    {
        free_.assign(R.host_pool->block_count(), 1);
        held_.assign(R.host_pool->block_count(), std::nullopt);
        fill_from_seed(R);
    }

    // Extend the view over freshly GROWN blocks — the
    // tail is free by construction; the existing rows (and any
    // in-acquire eviction bookkeeping the caller carries) stand.
    void grow_to(std::size_t new_count)
    {
        if (new_count <= free_.size())
        {
            return;
        }
        free_.resize(new_count, 1);
        held_.resize(new_count, std::nullopt);
    }

    void fill_from_seed(RuntimeState const &R)
    {
        std::size_t const count = free_.size();
        free_.assign(count, 1);
        held_.assign(count, std::nullopt);
        for (EntryBinding const &b : R.seed)
        {
            if (R.inv->is_host(b.node) &&
                static_cast<std::size_t>(b.chunk) < count)
            {
                free_[b.chunk] = 0;
                held_[b.chunk] = b;
            }
        }
    }

    std::size_t count() const { return free_.size(); }
};

// The acquire's device LANDING ladder (the boundary act's port of the
// planner's landing order, widened two rungs): (1) a FREE device
// chunk; (2) a BARE-SAFE device victim — the lowest device chunk
// whose binding is NOT the last actual replica of its logical chunk
// (an outdated binding holds no data; a replica-keeping binding's
// bytes survive on another node) — unlinked bare to BECOME the
// landing; (3) a last-actual device victim SWAPPED OUT to a free host
// block OUTSIDE the carve's window (its bytes land on the host — the
// drain direction — and the freed chunk becomes the landing; the tiny
// linear-step cap reaches this rung with every device chunk holding a
// last-actual replica). The world, the plan's queue, and the host
// blocks mutate together . Nullopt: no
// landing exists — the caller's census row.
// GROW FIRST for the boundary rescue: grow every
// device pool by `add` chunks (bounded by its soft limit), the full
// engine commit (pool bytes, inventory, the executor's applied table
// and dependency rows) — the fresh chunks are free, landings. Returns
// true when at least one chunk was added anywhere.
// One growth COMMIT for node n's pool,
// wherever a demand strikes (a plan's compile, an acquire's window, a
// rescue's landing): the pool's act ladder runs — a TILE-SIZED
// segment first, one chunk at a time when that does not work — and a
// pool the hard limit closed pins the inventory's ceiling to whatever
// it holds, so the planner's growth asks stop at the frozen budget
// and the LRU recycling order governs
// beyond it. Returns whether anything was added.
bool grow_pool_for(RuntimeState &R, NodeId n, std::size_t need)
{
    bool grew = false;
    if (R.inv->is_device(n))
    {
        CudaExecLane &lane = *R.exec_lanes.at(static_cast<std::size_t>(n));
        std::size_t const before = lane.cardinality();
        (void) lane.grow_to_wall(need);
        grew = lane.cardinality() > before;
        if (lane.hard_capped())
        {
            R.inv->close_growth(n, lane.cardinality());
        }
    }
    else
    {
        std::size_t const before = R.host_pool->block_count();
        (void) R.host_pool->grow_to_wall(need);
        grew = R.host_pool->block_count() > before;
        if (R.host_pool->growth_closed())
        {
            R.inv->close_growth(n, R.host_pool->block_count());
        }
    }
    return grew;
}

// The provisions follow the REAL slot space, whenever it
// moved: the ceiling's slot term (outstanding <= C_total * (1 + L))
// counts chunks HELD, so every growth commit — the plan
// commit, an acquire's window growth, a rescue's device growth — must
// re-sync the event provisions. A convergence that moves pools
// outside a plan's own growth vector (growth empty thereafter) is
// exactly the case the old per-plan extension missed.
void sync_event_provisions(RuntimeState &R, std::size_t plan_ops = 0)
{
    if (R.executor == nullptr)
    {
        return;
    }
    NodeInventory const &inv = *R.inv;
    std::size_t const nodes = inv.node_count();
    std::size_t const lane_count =
        nodes + 2 * (nodes - 1) +
        static_cast<std::size_t>(std::count_if(R.peer_bits.begin(),
            R.peer_bits.end(),
            [](std::uint8_t bit) { return bit != 0; }));
    std::size_t slots = 0;
    for (std::size_t n = 0; n < nodes; ++n)
    {
        slots += inv.pool_cardinality(static_cast<NodeId>(n));
    }
    std::size_t const ceiling = slots * (1 + lane_count);
    // Mints are per-op transients, not only per-held-chunk facts: a
    // long legal plan can mint about one event per op, so the plan's
    // op count is part of the sizing input.
    // Terminal events and late records add a few leases per lane; the
    // margin also covers the power-of-two boundary when plan_ops is
    // itself a power of two (two un-retired plans each mint ~plan_ops).
    std::size_t const provision = CudaEventPool::suggest_event_provision(
        {(ceiling + 1) / 2, plan_ops + lane_count + 4});
    for (std::unique_ptr<CudaEventPool> &pool : R.event_pools)
    {
        pool->extend(provision);
    }
    R.host_lane->extend_provision(provision);
}

bool grow_device_pools(RuntimeState &R, std::size_t add)
{
    // The rescue's landing demand (`add`
    // chunks) commits through the act ladder; pools at their wall or
    // closed add nothing.
    bool any = false;
    for (std::size_t d = 0; d < R.inv->device_count(); ++d)
    {
        NodeId const dev = static_cast<NodeId>(d);
        if (R.inv->pool_limit(dev) != 0
            && R.inv->pool_cardinality(dev) >= R.inv->pool_limit(dev))
        {
            continue; // at the soft wall already
        }
        if (!grow_pool_for(R, dev, add))
        {
            continue; // closed or nothing to add
        }
        R.inv->grow_node(dev, R.exec_lanes.at(d)->cardinality());
        if (R.executor != nullptr)
        {
            R.executor->apply_growth(dev, R.exec_lanes.at(d)->cardinality());
        }
        any = true;
    }
    if (any)
    {
        sync_event_provisions(R);
    }
    return any;
}

std::optional<std::pair<NodeId, ChunkId>> free_device_landing(
    RuntimeState const &R, SeedIndex &world, Plan &plan,
    HostBlocks &blocks, ChunkId window_base, std::size_t window_n)
{
    if (auto free_chunk = world.free_device_chunk(*R.inv);
        free_chunk.has_value())
    {
        return free_chunk;
    }
    for (std::size_t d = 0; d < R.inv->device_count(); ++d)
    {
        NodeId const dev = static_cast<NodeId>(d);
        auto const &seg = world.by_chunk_at(dev);
        for (std::size_t c = 0; c < R.inv->pool_cardinality(dev); ++c)
        {
            ChunkId const chunk = static_cast<ChunkId>(c);
            auto it = seg.find(chunk);
            if (it == seg.end())
            {
                return std::make_pair(dev, chunk); // the belt
            }
            LogicalChunk const lc{std::get<0>(it->second),
                std::get<1>(it->second)};
            bool const actual = std::get<2>(it->second);
            if (!actual ||
                world.actual_replica_elsewhere(lc.tile, lc.index, dev))
            {
                plan.queue.push_back(
                    make_unlink(lc.tile, lc.index, dev, chunk));
                world.unlink_binding(dev, chunk);
                return std::make_pair(dev, chunk);
            }
        }
    }
    // Rung 3: swap the lowest last-actual device chunk out to a free
    // host block outside the window, then land on it.
    NodeId const host = R.inv->host_id();
    for (std::size_t d = 0; d < R.inv->device_count(); ++d)
    {
        NodeId const dev = static_cast<NodeId>(d);
        auto const &seg = world.by_chunk_at(dev);
        for (std::size_t c = 0; c < R.inv->pool_cardinality(dev); ++c)
        {
            ChunkId const chunk = static_cast<ChunkId>(c);
            auto it = seg.find(chunk);
            if (it == seg.end())
            {
                return std::make_pair(dev, chunk); // the belt
            }
            LogicalChunk const lc{std::get<0>(it->second),
                std::get<1>(it->second)};
            for (std::size_t b = 0; b < blocks.count(); ++b)
            {
                if (!blocks.free_[b] ||
                    (b >= static_cast<std::size_t>(window_base) &&
                        b < static_cast<std::size_t>(window_base) +
                                window_n))
                {
                    continue; // occupied, or the carve's own run
                }
                ChunkId const host_block = static_cast<ChunkId>(b);
                plan.queue.push_back(
                    make_link(lc.tile, lc.index, host, host_block));
                plan.queue.push_back(make_copy(lc.tile, lc.index,
                    ChunkLoc{dev, chunk}, ChunkLoc{host, host_block}));
                plan.queue.push_back(
                    make_unlink(lc.tile, lc.index, dev, chunk));
                world.link_binding(
                    lc.tile, lc.index, host, host_block, true);
                world.unlink_binding(dev, chunk);
                EntryBinding landed;
                landed.node = host;
                landed.chunk = host_block;
                landed.tile = lc.tile;
                landed.index = lc.index;
                landed.actual = true;
                blocks.free_[b] = 0;
                blocks.held_[b] = landed;
                return std::make_pair(dev, chunk);
            }
            // The host pool is shared: if no free block outside the
            // window served this chunk, none serves any.
            return std::nullopt;
        }
    }
    return std::nullopt;
}

// The acquire's plan-time pool-exhaustion census (the
// the dead-end is NEVER a bare caller error from an engine act): the
// fault with the need, the world rendered per node with the pool
// bounds beside it, the entry state, the ops prefix built so far, and
// the inputs' relevant facts. The text lands on stderr beside every
// other census; the thrown nnhaul::Error keeps the acquire's API
// contract.
nnhaul::Error acquire_census(RuntimeState const &R, SeedIndex const &world,
    Plan const &plan, std::string const &need)
{
    Census c;
    c.fault = "pool-exhaustion";
    std::vector<EntryBinding> const all = world.bindings();
    for (std::size_t node = 0; node < R.inv->node_count(); ++node)
    {
        SegmentSnapshot seg;
        seg.node = static_cast<NodeId>(node);
        seg.pool = R.inv->pool_cardinality(static_cast<NodeId>(node));
        for (EntryBinding const &b : all)
        {
            if (b.node == node)
            {
                seg.bindings.push_back(b);
            }
        }
        c.fold.push_back(seg);
    }
    c.applied.clear();
    c.entry = plan.entry;
    c.prefix = plan.queue;
    c.inputs = "acquire plan_seq=" + std::to_string(plan.plan_seq) +
        "\nneed: " + need + "\n";
    std::string msg = "plan-time census: pool-exhaustion — " + need;
    std::fprintf(stderr, "nnhaul: %s\n", msg.c_str());
    return nnhaul::Error(msg);
}


// A census's payload rendered for the Error the API throws: the fault
// plus the fold/input lines .
std::string census_text(Census const &c)
{
    std::string s = c.fault;
    if (!c.inputs.empty())
    {
        s += "\n";
        s += c.inputs;
    }
    return s;
}

// The shared check-then-FUSED-execute tail of the engine's BOUNDARY
// plans (acquires, publishes, retirements — acts that need the bytes
// NOW, so their plans issue and join at once): the FULL checker inputs
// with the chain ALWAYS the current seed, the executor's fused
// issue+close, the seed advance to the checked exit, and the
// accounting. Requires R.mu.
// Commit a plan's growth BEFORE the check: grow the
// real pools (device lanes' arenas, the host pool — a hard-limit
// shortfall against the plan's ask is a loud abort, never a silent
// under-budget issue), the inventory, the executor's applied table and
// dependency rows, and the event provisions (the displacement ceiling
// grows with the slot space). The checker then validates the grown
// world; the executor's sparse boundary compare is unaffected (growth
// only ever appends FREE chunks). Requires R.mu (called from run_plan).
void apply_plan_growth(RuntimeState &R, Plan const &plan)
{
    NodeInventory const &inv = *R.inv;
    std::size_t const nodes = inv.node_count();
    for (std::size_t n = 0; n < nodes; ++n)
    {
        if (n >= plan.growth.size())
        {
            continue; // this plan grew nothing here
        }
        std::size_t const want = plan.growth[n];
        std::size_t const card =
            inv.pool_cardinality(static_cast<NodeId>(n));
        if (want > card)
        {
            // The plan's ask commits
            // through the act ladder (tile-sized first, one chunk at
            // a time when that does not work); the inventory and the
            // executor then sync to the POOL'S held count. The plan's
            // ops name ids < want — a wall that left the pool BELOW
            // the plan's real need is a loud abort (the budget is
            // fixed; there is no retry after a wall).
            grow_pool_for(R, static_cast<NodeId>(n), want - card);
            std::size_t const held_after =
                inv.is_device(static_cast<NodeId>(n))
                    ? R.exec_lanes.at(n)->cardinality()
                    : R.host_pool->block_count();
            if (held_after < want)
            {
                std::string const msg =
                    (inv.is_device(static_cast<NodeId>(n))
                         ? "nnhaul: plan growth of device " +
                             std::to_string(n) + " to " +
                             std::to_string(want) +
                             " chunks hit the wall at " +
                             std::to_string(held_after)
                         : "nnhaul: plan growth of the host pool to " +
                             std::to_string(want) +
                             " chunks hit the wall at " +
                             std::to_string(held_after)) +
                    " — the plan cannot be honored (the pool's budget "
                    "is fixed at its wall; lower the budget demand or "
                    "raise the soft limit)";
                R.set_error(msg);
                throw Error(msg);
            }
            std::size_t const target = held_after > card ? held_after
                                                         : card;
            R.inv->grow_node(static_cast<NodeId>(n), target);
            if (R.executor != nullptr)
            {
                R.executor->apply_growth(n, target);
            }
        }
    }
    // The provisions follow the slot space wherever it moved (and the
    // plan's op count, which bounds the transient mints).
    sync_event_provisions(R, plan.queue.size());
}

void run_plan(RuntimeState &R, Plan &plan,
    std::vector<SeqTask> const &seq)
{
    apply_plan_growth(R, plan); // grow before the check

    CheckInput in;
    in.plan = &plan;
    in.inventory = &*R.inv;
    in.tiles = &R.tiles;
    in.runnable = &R.runnable;
    in.task_sequence = &seq;
    in.chain = R.seed; // the executing pipeline supplies it always

    CheckResult verdict = check(in);
    if (!verdict.pass)
    {
        std::string msg =
            verdict.census.has_value()
                ? "check-time census: " + census_text(*verdict.census)
                : "check-time census: " + verdict.fault;
        std::fprintf(stderr, "nnhaul: %s\n", msg.c_str());
        R.set_error(msg);
        throw Error(msg);
    }

    // Retire dead tiles from the CHECKER's exit fold, not the planner's
    // prediction: the fold is the state the plan actually produced, so
    // the appended unlinks name bindings that exist. A planner exit that
    // omits or misbinds a dead binding cannot leak chunks this way. The
    // unlinks join the queue BEFORE execute(), which issues and closes in
    // one call: a trailing append would never be issued and the next
    // plan's entry check would fail.
    std::vector<HandleImpl *> swept;
    {
        std::vector<PlanOp> retire_ops;
        engine_retire_ops(R, verdict.exit, retire_ops, swept);
        if (!retire_ops.empty())
        {
            plan.queue.insert(plan.queue.end(), retire_ops.begin(),
                retire_ops.end());
            // The seed is the fold MINUS what just retired.
            auto &ex = verdict.exit;
            ex.erase(std::remove_if(ex.begin(), ex.end(),
                         [&](EntryBinding const &b) {
                             for (PlanOp const &op : retire_ops)
                             {
                                 if (op.kind == PlanOpKind::Unlink
                                     && op.u.unlink.node == b.node
                                     && op.u.unlink.chunk == b.chunk)
                                 {
                                     return true;
                                 }
                             }
                             return false;
                         }),
                ex.end());
        }
    }

    try
    {
        R.executor->execute(plan); // the fused form: issue + close
    }
    catch (ExecutorCensusError const &e)
    {
        std::string msg = std::string("execution census: ") + e.what();
        if (!e.census.inputs.empty())
        {
            msg += "\n";
            msg += e.census.inputs;
        }
        R.set_error(msg);
        throw Error(msg);
    }
    catch (std::exception const &e)
    {
        R.set_error(e.what());
        throw Error(e.what());
    }

    // The seed IS the checked exit fold; the duration model is
    // read-only from here on . A boundary plan is not a
    // user batch — no un-retired record, no plan_walks count.
    R.seed = verdict.exit;
    R.next_plan_seq = plan.plan_seq + 1;
    R.model_read_only = true;
    R.planned_ops += plan.queue.size();

    // execute() above is issue+close, so every appended UNLINK has run:
    // the retired handles leave the pending list now instead of being
    // swept again by every later plan until shutdown.
    for (HandleImpl *impl : swept)
    {
        engine_release_swept(R, impl);
    }

    engine_note_batch(R, plan, R.executor->last_issue());
    engine_simulate_plan(R, plan, R.executor->last_derivation());
}

// The pinned-run safety net: a prior plan evicted a chunk under a live
// Acquired — the pin's pointer story is broken; loud, never served.
// Requires R.mu.
void verify_pins(RuntimeState &R)
{
    SeedIndex idx(R.seed);
    NodeId const host = R.inv->host_id();
    for (auto const &kv : R.acquired)
    {
        HandleImpl const *impl = kv.first;
        AcquireRun const &run = kv.second;
        for (std::size_t i = 0; i < run.n; ++i)
        {
            auto b = idx.binding_of(
                impl->tile_id, static_cast<ChunkIndex>(i), host);
            if (!b.has_value() || b->first != run.base + i)
            {
                std::string msg =
                    "pinned acquire run of tile \"" + label_of(*impl) +
                    "\" chunk " + std::to_string(i) +
                    " lost its host binding — a plan evicted a pinned "
                    "acquire (release Acquired instances before "
                    "submitting work that evicts them)";
                R.set_error(msg);
                throw Error(msg);
            }
        }
    }
}

// The engine's host bodies : the gather reads nothing
// beyond its operands' addresses (the copies ARE the gather), and the
// publish writes nothing (the user's bytes already sit in the stable
// pool blocks; the TILE_OP is the publication).
void engine_host_noop(void *const *) noexcept
{
}

// The engine's own host-act symbols (the gather and the publish):
// interned once per run — a marker byte in the key's
// cl slot, the no-op body in the record. Requires R.mu.
TaskId engine_host_act_symbol(RuntimeState &R, unsigned char tag)
{
    SymbolKey key;
    key.codelet = 0;
    key.cl.assign(1, tag);
    auto it = R.symbol_of.find(key);
    if (it != R.symbol_of.end())
    {
        return it->second;
    }
    TaskId const id = R.next_symbol++;
    R.symbol_of.emplace(std::move(key), id);
    R.runnable.add(id, std::vector<NodeId>{R.inv->host_id()});
    auto reg = std::make_shared<HostTaskReg>();
    reg->engine_fn = &engine_host_noop;
    {
        std::lock_guard<std::mutex> lock(R.host_tasks_mu);
        R.host_tasks.emplace(id, std::move(reg));
    }
    R.symbol_registered.emplace(id, R.inv->host_id());
    R.host_act_id[tag] = id;
    return id;
}

} // namespace

// ---- the symbol interning  ----

// The O(1) key: equality over the same
// fields the old tree comparator walked, and one FNV-1a pass hashing
// them — the memoization lookup is now hash + (almost always single)
// equality probe instead of an O(log n) element-wise comparison walk.
bool SymbolKey::operator==(SymbolKey const &other) const
{
    return codelet == other.codelet && cl == other.cl
        && shapes == other.shapes;
}

std::size_t SymbolKeyHash::operator()(SymbolKey const &k) const
{
    std::size_t h = 1469598103934665603ULL;
    auto mix = [&h](void const *p, std::size_t n) {
        unsigned char const *b =
            static_cast<unsigned char const *>(p);
        for (std::size_t i = 0; i < n; ++i)
        {
            h ^= b[i];
            h *= 1099511628211ULL;
        }
    };
    mix(&k.codelet, sizeof(k.codelet));
    mix(k.cl.data(), k.cl.size());
    for (BufShape const &s : k.shapes)
    {
        mix(&s.n, sizeof(s.n));
        mix(&s.nbytes, sizeof(s.nbytes));
        mix(&s.access, sizeof(s.access));
    }
    return h;
}

std::size_t SymbolNodeHash::operator()(
    std::pair<TaskId, NodeId> const &p) const
{
    std::size_t h = 1469598103934665603ULL;
    auto mix = [&h](void const *q, std::size_t n) {
        unsigned char const *b = static_cast<unsigned char const *>(q);
        for (std::size_t i = 0; i < n; ++i)
        {
            h ^= b[i];
            h *= 1099511628211ULL;
        }
    };
    mix(&p.first, sizeof(p.first));
    mix(&p.second, sizeof(p.second));
    return h;
}

TaskId engine_intern_symbol(RuntimeState &R, Codelet &codelet,
    void const *cl_data, std::size_t cl_size,
    std::vector<BufShape> const &shapes)
{
    SymbolKey key;
    key.codelet = codelet.id();
    if (cl_size != 0 && cl_data != nullptr)
    {
        unsigned char const *p =
            static_cast<unsigned char const *>(cl_data);
        key.cl.assign(p, p + cl_size);
    }
    key.shapes = shapes;
    auto it = R.symbol_of.find(key);
    if (it != R.symbol_of.end())
    {
        return it->second; // memoized: never re-registered
    }
    TaskId const id = R.next_symbol++;
    // NNHAUL_SYMBOL_TRACE (diagnosis only): one line per new symbol,
    // the (codelet, cl_args hex, per-buffer chunk-count:bytes:access)
    // triple — the workload's op+shape census, off by default, never
    // control flow.
    std::string sym_hex;
    if (std::getenv("NNHAUL_SYMBOL_TRACE") != nullptr)
    {
        sym_hex.reserve(key.cl.size() * 2);
        for (unsigned char c : key.cl)
        {
            char buf[4];
            std::snprintf(buf, sizeof(buf), "%02x", c);
            sym_hex += buf;
        }
    }
    R.symbol_of.emplace(std::move(key), id);
    // The executability row and the duration key map the SYMBOL (one
    // row per symbol family, not per instance).
    R.runnable.add(id, runnable_nodes_of(codelet, *R.inv));
    R.task_codelet.emplace(id, codelet.id());
    if (!sym_hex.empty())
    {
        static const char access_char[3] = {'r', 'w', 'b'};
        std::fprintf(stderr, "[nnhaul-symbol] id=%u codelet=%s cl=%s "
                     "bufs=", static_cast<unsigned>(id),
            codelet.name().c_str(), sym_hex.c_str());
        for (BufShape const &s : shapes)
        {
            std::fprintf(stderr, "%zu:%zu:%c ", s.n, s.nbytes,
                access_char[static_cast<int>(s.access)]);
        }
        std::fprintf(stderr, "\n");
    }
    return id;
}

void engine_ensure_symbol_registered(RuntimeState &R, TaskId symbol,
    NodeId node, Codelet &codelet, void const *cl_data,
    std::size_t cl_size, std::vector<BufShape> const &shapes)
{
    if (!R.symbol_registered.emplace(symbol, node).second)
    {
        return; // one registration per (symbol, node), forever
    }
    {
        // The breakdown histogram's name (once per symbol; the map
        // node's string outlives every body that points at it).
        std::lock_guard<std::mutex> lock(R.body_mu);
        R.symbol_name.emplace(symbol,
            &(R.codelet_names.emplace(codelet.id(),
                 codelet.name())
                   .first->second));
        R.body_breakdown_.store(
            std::getenv("NNHAUL_ISSUE_BREAKDOWN") != nullptr,
            std::memory_order_relaxed);
    }
    if (R.inv->is_host(node))
    {
        auto reg = std::make_shared<HostTaskReg>();
        reg->codelet = &codelet;
        reg->shapes = shapes;
        if (cl_size != 0 && cl_data != nullptr)
        {
            unsigned char const *p =
                static_cast<unsigned char const *>(cl_data);
            reg->cl.assign(p, p + cl_size);
        }
        {
            std::lock_guard<std::mutex> lock(R.host_tasks_mu);
            R.host_tasks.emplace(symbol, std::move(reg));
        }
        return;
    }
    CudaExecLane *lane = R.exec_lanes[node].get();
    std::string const *body_name = nullptr;
    {
        std::lock_guard<std::mutex> lock(R.body_mu);
        auto const it = R.symbol_name.find(symbol);
        if (it != R.symbol_name.end())
        {
            body_name = it->second;
        }
    }
    auto body = engine_device_body(R, node, codelet, cl_data, cl_size,
        shapes, body_name);
    if (R.executor != nullptr &&
        R.executor->deferral_trailing(
            static_cast<LaneId>(node))) // exec(node) == node
    {
        // THIS lane's queue still trails an un-joined plan's steps:
        // the registration rides the SAME queue (it precedes every
        // launch of this symbol in the lane's issue order), so the
        // lane's waiter never races the table insert. Other lanes
        // register directly — the per-lane re-scope.
        R.bridge->submit(static_cast<LaneId>(node), 0,
            [lane, symbol, body = std::move(body)]() mutable {
                lane->register_task(symbol, std::move(body));
            });
        return;
    }
    lane->register_task(symbol, std::move(body));
}

// ---- the registration records  ----

CudaLaunchFn engine_device_body(RuntimeState &R, NodeId node,
    Codelet &codelet, void const *cl_data, std::size_t cl_size,
    std::vector<BufShape> shapes,
    std::string const *body_name)
{
    DeviceFacts const facts = R.dev[node];
    Codelet::Kernel const body = codelet.cuda_;
    std::vector<unsigned char> cl;
    if (cl_size != 0 && cl_data != nullptr)
    {
        cl.assign(static_cast<unsigned char const *>(cl_data),
            static_cast<unsigned char const *>(cl_data) + cl_size);
    }
    RuntimeState *const rp = &R;
    return [rp, node, facts, body, cl = std::move(cl),
               shapes = std::move(shapes),
               body_name](cudaStream_t stream, void *const *addrs) {
        // The RUNNING node's TLS, set at launch
        // dispatch and valid only inside the codelet window.
        Tls const save = tls;
        tls = Tls{};
        tls.in_kernel = true;
        tls.host_node = false;
        tls.node = node;
        tls.stream = stream;
        tls.cublas = facts.cublas;
        tls.workspace = facts.workspace;
        tls.workspace_bytes = facts.workspace_bytes;
        tls.copy_desc = facts.copy_desc;
        tls.copy_desc_bytes = facts.copy_desc_bytes;
        tls.device = facts.ordinal;

        // The Buffer views: one per task buffer, the symbol's shapes
        // giving each its chunk span in the TILE_OP's operand order
        // (per buffer, indexes ascending).
        Buffer inline_b[Task::kInlineBufs];
        void *inline_p[Task::kInlineBufs];
        std::vector<Buffer> heap_b;
        std::vector<void *> heap_p;
        std::size_t nbufs = shapes.size();
        bool const heap = nbufs > Task::kInlineBufs;
        if (heap)
        {
            heap_b.resize(nbufs);
            heap_p.resize(nbufs);
        }
        std::size_t at = 0;
        for (std::size_t j = 0; j < nbufs; ++j)
        {
            BufShape const &shape = shapes[j];
            Buffer &b = heap ? heap_b[j] : inline_b[j];
            b.ptr_ = addrs[at];
            b.capacity_ = shape.n * chunk_class_bytes();
            b.nbytes_ = shape.nbytes;
            b.chunk_ptrs_ = shape.n > 1 ? addrs + at : nullptr;
            b.chunk_count_ = shape.n;
            b.chunk_bytes_ = shape.n > 1 ? chunk_class_bytes() : 0;
            b.host_ptr_ = nullptr; // no host replica beside a device op
            (heap ? heap_p[j] : inline_p[j]) = &b;
            at += shape.n;
        }
        rp->kernel_launches_.fetch_add(1, std::memory_order_relaxed);
        if (rp->body_breakdown_.load(std::memory_order_relaxed))
        {
            // The breakdown knob's bracket (diagnosis only): the
            // whole body — the launch/cuBLAS host work this thread
            // pays — attributed to the codelet's name.
            std::uint64_t const t0 = now_ns();
            body(heap ? heap_p.data() : inline_p,
                cl.empty() ? nullptr
                    : const_cast<unsigned char *>(cl.data()));
            std::uint64_t const dt = now_ns() - t0;
            tls = save;
            std::lock_guard<std::mutex> lock(rp->body_mu);
            auto &slot = rp->body_ns[body_name != nullptr
                         ? *body_name
                         : std::string("?")];
            slot.first += dt;
            slot.second += 1;
            return;
        }
        body(heap ? heap_p.data() : inline_p,
            cl.empty() ? nullptr
                : const_cast<unsigned char *>(cl.data()));
        tls = save;
    };
}

LaunchResolver engine_host_resolver(RuntimeState &R)
{
    RuntimeState *const rp = &R;
    return [rp](TaskId t, void *const *addrs) {
        // The symbol's IMMUTABLE record, served under the registration
        // mutex — the engine may be interning the next batch's symbols
        // while this launch serves (the pipelined discipline).
        std::shared_ptr<HostTaskReg const> reg;
        {
            std::lock_guard<std::mutex> lock(rp->host_tasks_mu);
            auto it = rp->host_tasks.find(t);
            if (it != rp->host_tasks.end())
            {
                reg = it->second;
            }
        }
        if (reg == nullptr)
        {
            throw std::runtime_error(
                "host lane: task " + std::to_string(t) +
                " has no registered symbol — identity resolves at "
                "registration ");
        }
        Tls const save = tls;
        tls = Tls{};
        tls.in_kernel = true;
        tls.host_node = true;
        tls.node = rp->inv->host_id();

        Buffer inline_b[Task::kInlineBufs];
        void *inline_p[Task::kInlineBufs];
        std::vector<Buffer> heap_b;
        std::vector<void *> heap_p;
        std::size_t const nbufs = reg->shapes.size();
        bool const heap = nbufs > Task::kInlineBufs;
        if (heap)
        {
            heap_b.resize(nbufs);
            heap_p.resize(nbufs);
        }
        std::size_t at = 0;
        for (std::size_t j = 0; j < nbufs; ++j)
        {
            BufShape const &shape = reg->shapes[j];
            Buffer &b = heap ? heap_b[j] : inline_b[j];
            b.ptr_ = addrs[at];
            b.capacity_ = shape.n * chunk_class_bytes();
            b.nbytes_ = shape.nbytes;
            b.chunk_ptrs_ = shape.n > 1 ? addrs + at : nullptr;
            b.chunk_count_ = shape.n;
            b.chunk_bytes_ = shape.n > 1 ? chunk_class_bytes() : 0;
            // The host node's own op: the operand IS the host replica.
            b.host_ptr_ = shape.n == 1 ? addrs[at] : nullptr;
            (heap ? heap_p[j] : inline_p[j]) = &b;
            at += shape.n;
        }
        if (reg->engine_fn != nullptr)
        {
            reg->engine_fn(addrs);
        }
        else
        {
            if (reg->codelet == nullptr || reg->codelet->cpu_ == nullptr)
            {
                throw std::runtime_error(
                    "host lane: symbol " + std::to_string(t) +
                    " carries no CPU body — task-not-runnable closes "
                    "the class at check time ; this refusal is "
                    "the residual backstop");
            }
            reg->codelet->cpu_(heap ? heap_p.data() : inline_p,
                reg->cl.empty()
                    ? nullptr
                    : const_cast<unsigned char *>(reg->cl.data()));
        }
        tls = save;
    };
}

// The batch's symbol pass: every pending task interns its (codelet,
// cl_args, shapes) symbol — memoized, so the second sighting of one
// triple registers nothing — and ensures the per-node registration.
// The pass body lives INLINE in engine_issue_batch's fused
// register+request walk (one pointer-chase of the pending chain, the
// giant single-shot plan's dominant registration cost); this comment
// documents it. Requires R.mu.
namespace
{
// NNHAUL_SYMBOL_TRACE's per-symbol instance tally (diagnosis only; the
// process has one RuntimeState, this static is its tally).
std::unordered_map<std::uint32_t, std::uint64_t> g_symbol_count;
}

// ---- init  and shutdown  ----

void engine_init(int ncpu, int ncuda, std::size_t cpu_cap_bytes,
    std::size_t cuda_cap_bytes_each)
{

    RuntimeState &R = *rt;
    warn_retired_knobs();
    {
        static std::atomic<std::uint64_t> epoch{0};
        R.generation = epoch.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    g_symbol_count.clear(); // per-runtime diagnostic, not process history

    // ncpu is a compatibility parameter — the host node
    // exists exactly once — accepting exactly 0 or 1, else loud.
    if (ncpu != 0 && ncpu != 1)
    {
        throw Error("nnhaul::init: ncpu of " + std::to_string(ncpu) +
                    " is neither 0 nor 1 — the host node exists exactly "
                    "once ");
    }

    // Step 1: select and probe — the run's ONLY device introspection
    // .
    R.ordinals = select_devices(ncuda);
    R.peer_bits = probe_peer_matrix(R.ordinals);

    // Step 2: contexts — each selected device's primary context made
    // current (the lane constructors re-bind as they create streams;
    // this step owns the INITIALIZATION itself).
    for (int ordinal : R.ordinals)
    {
        cudaError_t err = cudaSetDevice(ordinal);
        if (err != cudaSuccess)
        {
            throw Error(std::string("nnhaul::init: cudaSetDevice(") +
                        std::to_string(ordinal) +
                        "): " + cudaGetErrorString(err));
        }
    }

    // Step 3: peer-enable — after the contexts, on the probed pairs.
    enable_peer_access(R.ordinals, R.peer_bits);

    // Step 7 before the lanes it provisions (construction order, not
    // the logical numbering): the per-device event pools
    // must exist before the lanes bind them. M_p is FORWARD-LOOKING at
    // init — no plan exists yet — so the provision sizes to the one
    // init-time bound: the DISPLACEMENT CEILING,
    // outstanding <= C_total * (1 + L) (the registry holds an entry
    // while any USER/PRODUCER fact names its op, and the USER fact
    // never resets — a run with many live (chunk, lane)
    // facts legitimately holds thousands of entries at once; the old
    // static 1024 floor was exhausted exactly here). The
    // cardinalities and the lane count are plain numbers before the
    // inventory object exists: provision = max(1024,
    // pow2ceil(C_total * (1 + L))), the formula floor retained,
    // NNHAUL_EVENT_POOL still overriding. The host lane's flag array
    // takes the same number. Exhaustion stays
    // a loud configuration fault.
    // Fixed per-device workspaces, allocated BESIDE the arena: the
    // GEMM gather/scatter workspace (the knob below), the cuBLAS
    // workspace, and the 1-MiB copy-descriptor scratch. They are
    // claimed after the arena, so an uncapped arena sizing must leave
    // room for them (suggest_arena_bytes subtracts this reserve);
    // otherwise the arena's own free-VRAM estimate starves the
    // workspace cudaMalloc.
    std::size_t workspace_bytes = std::size_t{1024} << 20;
    {
        char const *ws_env = std::getenv("NNHAUL_WORKSPACE_MB");
        if (ws_env != nullptr && *ws_env != '\0')
        {
            long ws_mib = 0;
            if (!env_parse_long(ws_env, ws_mib) || ws_mib <= 0)
            {
                std::fprintf(stderr,
                    "[nnhaul] NNHAUL_WORKSPACE_MB=%s is not a positive "
                    "integer; using 1024\n",
                    ws_env);
            }
            else if (static_cast<unsigned long>(ws_mib)
                > std::numeric_limits<std::size_t>::max() >> 20)
            {
                throw Error(std::string("nnhaul: NNHAUL_WORKSPACE_MB=")
                    + ws_env + " is too large to convert to bytes");
            }
            else
            {
                workspace_bytes =
                    static_cast<std::size_t>(ws_mib) << 20;
            }
        }
    }
    std::size_t const cublas_ws_bytes = cublas_workspace_bytes();
    std::size_t const copy_desc_bytes = std::size_t{1} << 20; // 1 MiB
    std::size_t const fixed_reserve_bytes =
        workspace_bytes + cublas_ws_bytes + copy_desc_bytes;
    (void)fixed_reserve_bytes; // the reserve shaped the old
                               // upfront carve; the two-phase ladder
                               // claims tile-sized acts and stops at
                               // the non-allocable flip

    // Limits and initials: every pool starts at its
    // explicit PREALLOC head start (NNHAUL_PREALLOC_CUDA_MEM per
    // device, NNHAUL_PREALLOC_CPU_MEM for the host — each CHECKED
    // against its soft limit, a larger ask a loud init fault; default
    // 0 = nothing) and grows at planning time through the two-phase
    // ladder (tile-wise acts, single-chunk fallback per event, the
    // non-allocable flip) to its soft limit — the env knob / init
    // argument ceiling.
    auto prealloc_blocks = [](char const *knob) -> std::size_t {
        std::size_t const mib = prealloc_mib(knob);
        return mib == 0 ? 0
                        : std::max<std::size_t>(
                              1, (mib << 20) / chunk_class_bytes());
    };
    std::size_t const prealloc_cuda = prealloc_blocks(
        "NNHAUL_PREALLOC_CUDA_MEM");
    std::size_t const prealloc_cpu =
        prealloc_blocks("NNHAUL_PREALLOC_CPU_MEM");
    std::vector<std::size_t> cardinalities;
    cardinalities.reserve(R.ordinals.size() + 1);
    std::vector<std::size_t> limits;
    limits.reserve(R.ordinals.size() + 1);
    for (std::size_t d = 0; d < R.ordinals.size(); ++d)
    {
        std::size_t const limit_bytes = CudaExecLane::suggest_limit_bytes(
            R.ordinals[d], cuda_cap_bytes_each, fixed_reserve_bytes);
        std::size_t const limit =
            limit_bytes == 0 ? 0 : limit_bytes / chunk_class_bytes();
        // Preallocation is an INIT act,
        // checked against the limit: a head start above it is a loud
        // configuration fault, never clamped.
        if (limit != 0 && prealloc_cuda > limit)
        {
            throw Error(
                "nnhaul: NNHAUL_PREALLOC_CUDA_MEM exceeds device " +
                std::to_string(d) + "'s soft limit (" +
                std::to_string(prealloc_cuda) + " > " +
                std::to_string(limit) + " chunks)");
        }
        limits.push_back(limit);
        cardinalities.push_back(prealloc_cuda);
    }
    std::size_t const host_limit =
        HostPool::resolve_limit_blocks(cpu_cap_bytes);
    if (host_limit != 0 && prealloc_cpu > host_limit)
    {
        throw Error(
            "nnhaul: NNHAUL_PREALLOC_CPU_MEM exceeds the host pool's "
            "soft limit (" + std::to_string(prealloc_cpu) + " > " +
            std::to_string(host_limit) + " blocks)");
    }
    limits.push_back(host_limit);
    cardinalities.push_back(prealloc_cpu);
    std::size_t const lane_count = (R.ordinals.size() + 1) + // exec lanes
        2 * R.ordinals.size() +                              // host chans
        static_cast<std::size_t>(std::count_if(R.peer_bits.begin(),
            R.peer_bits.end(),
            [](std::uint8_t bit) { return bit != 0; })); // peer chans
    std::size_t slots = 0;
    for (std::size_t card : cardinalities)
    {
        slots += card;
    }
    std::size_t const ceiling = slots * (1 + lane_count);
    std::size_t const provision =
        CudaEventPool::suggest_event_provision({(ceiling + 1) / 2});
    for (std::size_t d = 0; d < R.ordinals.size(); ++d)
    {
        R.event_pools.push_back(std::make_unique<CudaEventPool>(
            R.ordinals[d], provision));
    }

    // Step 4: arenas — segmented per chunk (the lane owns the
    // cudaMallocs; its stream rides along, a backend-internal detail
    // of the landed CudaExecLane). Initial = the prealloc head start.
    for (std::size_t d = 0; d < R.ordinals.size(); ++d)
    {
        R.exec_lanes.push_back(std::make_unique<CudaExecLane>(
            R.ordinals[d], limits[d], cardinalities[d], *R.event_pools[d]));
    }

    // Step 5: the host pool — segmented pinned blocks, the same
    // growth pattern.
    R.host_pool = std::make_unique<HostPool>(
        host_limit, cardinalities.back());
    // One census line at init: the sizing facts an OOM or a cap bug
    // investigation needs (the initials and limits), stated loudly.
    std::string const prealloc_note =
        std::string("cpu prealloc ") +
        (prealloc_cpu == 0 ? std::string("off")
                           : std::to_string(prealloc_cpu) + " blocks") +
        ", cuda prealloc " +
        (prealloc_cuda == 0 ? std::string("off")
                            : std::to_string(prealloc_cuda) +
                                  " chunks/device");
    std::fprintf(stderr,
        "[nnhaul] init: %zu device(s), arena %zu/%s chunks, host pool "
        "%zu/%s blocks, %s\n",
        R.ordinals.size(),
        cardinalities.empty() ? std::size_t{0} : cardinalities[0],
        limits.empty()
            ? "?"
            : (limits[0] == 0
                  ? std::string("unlimited")
                  : std::to_string(limits[0]))
                .c_str(),
        cardinalities.back(),
        host_limit == 0 ? "unlimited"
                        : std::to_string(host_limit).c_str(),
        prealloc_note.c_str());

    // The inventory: composed once the cardinalities exist (the lane
    // set, the channel matrix, the pool bounds), now with
    // the pools' soft limits for the planner's growth.
    R.inv = std::make_unique<NodeInventory>(
        R.ordinals.size(), cardinalities, R.peer_bits, limits);
    // An external lowering planner, when NNHAUL_PLANNER names one. It is
    // loaded once and lives as long as the process; the in-tree LRU path
    // is used when the variable is unset.
    R.planner_plugin = planner::load_from_env();

    // Step 6: lanes — the host worker, the deferral bridge over it,
    // then the transfer channels per connected ordered pair (the
    // executor receives them dense in channel-lane id order, which is
    // the inventory's ascending (a, b) numbering — the same walk
    // below), the bridge wired into EVERY CUDA lane.
    R.host_lane = std::make_unique<HostWorkerLane>(
        provision, engine_host_resolver(R));
    R.bridge = std::make_unique<DeferralBridge>(*R.host_lane);
    R.default_cuda_node = 0;
    R.default_cpu_node = R.inv->host_id();
    for (std::size_t a = 0; a < R.inv->node_count(); ++a)
    {
        for (std::size_t b = 0; b < R.inv->node_count(); ++b)
        {
            if (a == b || !R.inv->has_channel(static_cast<NodeId>(a),
                                          static_cast<NodeId>(b)))
            {
                continue;
            }
            // The host node has no CUDA ordinal; R.ordinals is indexed
            // by device node only.
            int const ordinal_a = R.inv->is_host(static_cast<NodeId>(a))
                ? -1
                : R.ordinals[a];
            if (R.inv->is_host(static_cast<NodeId>(a)))
            {
                // host -> device: the stream on the pair's single
                // device .
                auto chan = std::make_unique<CudaHostChannel>(
                    R.ordinals[b], HostChannelDirection::HostToDevice,
                    *R.event_pools[b]);
                R.channels.push_back(std::move(chan));
            }
            else if (R.inv->is_host(static_cast<NodeId>(b)))
            {
                auto chan = std::make_unique<CudaHostChannel>(
                    ordinal_a, HostChannelDirection::DeviceToHost,
                    *R.event_pools[a]);
                R.channels.push_back(std::move(chan));
            }
            else
            {
                auto chan = std::make_unique<CudaPeerChannel>(ordinal_a,
                    R.ordinals[b], *R.event_pools[b]);
                R.channels.push_back(std::move(chan));
            }
        }
    }

    // Step 8: per-device workspaces, library handles .
    // Codelet registration per node is the MEMOIZED TASK SYMBOLS of
    // engine_intern_symbol — one registration per (symbol, node) at
    // the symbol's first sight mid-run (the workload's inserts are the
    // registration surface; codelets outlive the runtime); the tables
    // live on R. The gather/scatter workspace knob, the cuBLAS
    // workspace knob, and the 1-MiB copy-descriptor scratch are per
    // device, beside the pool, never chunks .
    // The workspaces themselves (sizes resolved above, before the
    // arena, so the uncapped sizing could reserve them).
    R.dev.resize(R.ordinals.size());
    // A mid-step failure must not leak the fixed workspaces already
    // carved (engine_shutdown is the runtime's own act; a failed init
    // unwinds here) — free what exists before the rethrow.
    try
    {
    for (std::size_t d = 0; d < R.ordinals.size(); ++d)
    {
        DeviceFacts &f = R.dev[d];
        f.ordinal = R.ordinals[d];
        cudaError_t err = cudaSetDevice(f.ordinal);
        if (err != cudaSuccess)
        {
            throw Error(std::string("nnhaul::init: cudaSetDevice(") +
                        std::to_string(f.ordinal) +
                        "): " + cudaGetErrorString(err));
        }
        f.workspace_bytes = workspace_bytes;
        err = cudaMalloc(&f.workspace, f.workspace_bytes);
        if (err != cudaSuccess)
        {
            throw Error("workspace cudaMalloc of " +
                        std::to_string(f.workspace_bytes) +
                        " bytes on device " + std::to_string(f.ordinal) +
                        " failed: " + cudaGetErrorString(err));
        }
        f.alloc_count += 1;
        f.copy_desc_bytes = copy_desc_bytes; // 1 MiB, as today
        err = cudaMalloc(&f.copy_desc, f.copy_desc_bytes);
        if (err != cudaSuccess)
        {
            throw Error("chunk-copy descriptor cudaMalloc on device " +
                        std::to_string(f.ordinal) + " failed: " +
                        cudaGetErrorString(err));
        }
        f.alloc_count += 1;
        cublasStatus_t blas_err = cublasCreate(&f.cublas);
        if (blas_err != CUBLAS_STATUS_SUCCESS)
        {
            throw Error("cublasCreate failed on device " +
                        std::to_string(f.ordinal));
        }
        // The handle is bound to that device's
        // execution lane's stream.
        if (cublasSetStream(f.cublas, R.exec_lanes[d]->stream()) !=
            CUBLAS_STATUS_SUCCESS)
        {
            throw Error("cublasSetStream failed on device " +
                        std::to_string(f.ordinal));
        }
        if (cublas_ws_bytes > 0)
        {
            f.cublas_workspace_bytes = cublas_ws_bytes;
            err = cudaMalloc(&f.cublas_workspace, cublas_ws_bytes);
            if (err != cudaSuccess)
            {
                throw Error("cublas workspace cudaMalloc on device " +
                            std::to_string(f.ordinal) + ": " +
                            cudaGetErrorString(err));
            }
            f.alloc_count += 1;
            if (cublasSetWorkspace(f.cublas, f.cublas_workspace,
                    cublas_ws_bytes) != CUBLAS_STATUS_SUCCESS)
            {
                throw Error("cublasSetWorkspace failed on device " +
                            std::to_string(f.ordinal));
            }
        }
    }
    }
    catch (...)
    {
        for (std::size_t d = 0; d < R.dev.size(); ++d)
        {
            DeviceFacts &f = R.dev[d];
            (void) cudaSetDevice(f.ordinal);
            if (f.cublas != nullptr)
            {
                cublasDestroy(f.cublas);
            }
            if (f.cublas_workspace != nullptr)
            {
                (void) cudaFree(f.cublas_workspace);
            }
            if (f.workspace != nullptr)
            {
                (void) cudaFree(f.workspace);
            }
            if (f.copy_desc != nullptr)
            {
                (void) cudaFree(f.copy_desc);
            }
        }
        throw;
    }

    // The tracer (armed by NNHAUL_TRACE) and the executor over the
    // lanes, pools, addressers, and the bridge. The stamps ride the
    // lanes' own provisions (the stamp provision) — sized by no engine
    // arithmetic: an untraced run allocates nothing, a traced run's
    // provision grows to the working set and recycles at the
    // harvests . Both borrow R.tiles — the tile table
    // the COPY spans derive from (the executor
    // issues it, the tracer stamps it); the table's rows join at
    // handle registration below, before any plan runs.
    R.tracer = std::make_unique<PlanTrace>(*R.inv, R.tiles);
    if (auto path = PlanTrace::knob_path())
    {
        R.tracer->arm(*path);
    }
    {
        std::vector<ExecLane *> execs;
        execs.reserve(R.inv->node_count());
        for (std::size_t d = 0; d < R.exec_lanes.size(); ++d)
        {
            execs.push_back(R.exec_lanes[d].get());
        }
        execs.push_back(R.host_lane.get());
        std::vector<TransferChannel *> chans;
        chans.reserve(R.channels.size());
        for (auto &c : R.channels)
        {
            chans.push_back(c.get());
        }
        std::vector<EventPool *> pools;
        pools.reserve(R.exec_lanes.size());
        for (auto &p : R.event_pools)
        {
            pools.push_back(p.get());
        }
        std::vector<ChunkAddresser> addr;
        std::vector<ChunkContig> contig;
        addr.reserve(R.inv->node_count());
        for (std::size_t d = 0; d < R.exec_lanes.size(); ++d)
        {
            CudaExecLane *lane = R.exec_lanes[d].get();
            addr.push_back([lane](ChunkId c) { return lane->addr(c); });
            // The chunk's SEGMENT extent: a merged copy
            // may span the segment's VA-contiguous chunks, never
            // past its allocation.
            contig.push_back([lane](ChunkId c) {
                return lane->contig_bytes(c);
            });
        }
        HostPool *pool = R.host_pool.get();
        addr.push_back([pool](ChunkId c) { return pool->addr(c); });
        contig.push_back([pool](ChunkId c) {
            return pool->contig_bytes(c);
        });
        // The twins' seam: a dedicated EMPTY stream
        // on device 0 — the host worker's record twin is a
        // cudaEventRecord on it AT the mint's FIFO position, so the
        // event's completion IS that position (nothing else precedes
        // it on the stream) and device consumers wait stream-side.
        // The hook's failure path faults the host lane (the worker's
        // own catch) AND releases the bridge's recorded-wait parks
        // (the twin's deadlock closure).
        {
            // The twin stream and its events live on device 0's ordinal.
            // The init loop leaves the LAST selected ordinal current, and
            // the host worker thread starts on device 0, so pin the
            // ordinal for the stream creation and restore the caller.
            int prev_device = -1;
            (void)cudaGetDevice(&prev_device);
            (void)cudaSetDevice(R.ordinals.at(0));
            cudaError_t const err = cudaStreamCreateWithFlags(
                &R.host_mint_stream, cudaStreamNonBlocking);
            (void)cudaSetDevice(prev_device);
            if (err != cudaSuccess)
            {
                throw Error(std::string("init: the host-mint twins' "
                                        "record stream: ") +
                            cudaGetErrorString(err));
            }
        }
        R.executor = std::make_unique<PlanExecutor>(*R.inv,
            std::move(execs), R.host_lane.get(), std::move(chans),
            std::move(pools), std::move(addr), std::move(contig),
            R.bridge.get(),
            R.event_pools.at(0).get(),
            [&R](EventHandle e) {
                // The record runs on the host worker thread, whose current
                // device is not necessarily the table's device 0 ordinal.
                (void)cudaSetDevice(R.ordinals.at(0));
                cudaError_t const err = cudaEventRecord(
                    R.event_pools.at(0)->raw(e), R.host_mint_stream);
                if (err != cudaSuccess)
                {
                    R.bridge->release_recorded_waits(
                        "the host-mint record twin was refused: " +
                        std::string(cudaGetErrorString(err)));
                    throw std::runtime_error(
                        std::string("host-mint record twin: ") +
                        cudaGetErrorString(err));
                }
            },
            R.tiles,
            R.tracer.get());
    }

    // Step 9: the binding table (the executor's applied table) is all
    // free; the initial load is EMPTY — no tile holds bytes before the
    // first plan (the drained initial world), so the seed is empty and
    // the first plan's chain input is the empty state.
    R.seed = R.executor->applied().snapshot();

    // Handles constructed before init join the tile table now.
    for (HandleImpl *impl : handle_registry_snapshot())
    {
        R.tiles.add(impl->tile_id, impl->nbytes);
        R.live_tiles.emplace(impl->tile_id, impl);
    }
}

void echo_counters_at_shutdown(RuntimeState const &R)
{
    // NNHAUL_ALLOC_TRACE's one meaning (the header's comment): the
    // counters echo, read from the same state the test hooks read —
    // kernel_launches_ and the ManagerStats counters on R, plus the
    // executor's ops_issued and per-plan reconcile witnesses (the
    // executor must still be alive; the caller runs this before its
    // reset). One line, stable key=value pairs, nothing else.
    if (std::getenv("NNHAUL_ALLOC_TRACE") == nullptr)
    {
        return;
    }
    std::uint64_t ops_issued = 0;
    bool reconcile_ok = true;
    if (R.executor != nullptr)
    {
        ops_issued = R.executor->ops_issued();
        for (PlanExecutor::PlanIssue const &issue :
            R.executor->issues())
        {
            reconcile_ok = reconcile_ok && issue.reconcile_ok;
        }
    }
    std::fprintf(stderr,
        "[nnhaul-counters] kernel_launches=%llu plan_walks=%llu "
        "ops_planned=%llu ops_issued=%llu reconcile_ok=%d "
        "copy_ops=%llu tile_ops=%llu link_ops=%llu unlink_ops=%llu "
        "cross_node_copies=%llu evictions=%llu rescues=%llu "
        "refetches=%llu prefetches=%llu dying_frees=%llu "
        "waiter_busy_ms=%.1f "
        "waiter_parked_ms=%.1f waiter_steps=%llu gates=%llu "
        "recwaits=%llu close_join_ms=%.1f close_drain_ms=%.1f "
        "close_host_ms=%.1f close_total_ms=%.1f host_launch_ms=%.1f "
        "host_launches=%llu\n",
        static_cast<unsigned long long>(
            R.kernel_launches_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(R.batches_planned),
        static_cast<unsigned long long>(R.planned_ops),
        static_cast<unsigned long long>(ops_issued),
        reconcile_ok ? 1 : 0,
        static_cast<unsigned long long>(R.copy_ops),
        static_cast<unsigned long long>(R.tile_ops),
        static_cast<unsigned long long>(R.link_ops),
        static_cast<unsigned long long>(R.unlink_ops),
        static_cast<unsigned long long>(R.copy_log_.size()),
        static_cast<unsigned long long>(R.evictions),
        static_cast<unsigned long long>(R.rescues),
        static_cast<unsigned long long>(R.refetches),
        static_cast<unsigned long long>(R.prefetches),
        static_cast<unsigned long long>(R.dying_frees),
        static_cast<double>(R.bridge->waiter_busy_ns()) / 1.0e6,
        static_cast<double>(R.bridge->waiter_parked_ns()) / 1.0e6,
        static_cast<unsigned long long>(R.bridge->waiter_steps()),
        static_cast<unsigned long long>(R.bridge->gates_parked()),
        static_cast<unsigned long long>(R.bridge->record_waits()),
        static_cast<double>(R.executor ? R.executor->close_join_ns()
                                       : 0) /
            1.0e6,
        static_cast<double>(R.executor ? R.executor->close_drain_ns()
                                       : 0) /
            1.0e6,
        static_cast<double>(R.executor ? R.executor->close_host_ns()
                                       : 0) /
            1.0e6,
        static_cast<double>(R.executor ? R.executor->close_total_ns()
                                       : 0) /
            1.0e6,
        static_cast<double>(R.host_lane->launch_busy_ns()) / 1.0e6,
        static_cast<unsigned long long>(R.host_lane->launch_count()));
    // The gate-family histogram (the executor's diagnosis line): the
    // waited host mint's task symbol per device-waits-host gate. The
    // engine's own host-act symbols print beside it so the families
    // read by name, not by id.
    if (R.executor != nullptr && R.executor->gates_total() != 0)
    {
        std::string rows;
        for (auto const &kv : R.executor->gate_waited_tasks())
        {
            rows += " t" + std::to_string(kv.first) + "x" +
                std::to_string(kv.second);
        }
        std::string consumers;
        for (auto const &kv : R.executor->gate_consumers())
        {
            consumers += " " +
                std::string(kv.first >= 100 ? "same:" : "cross:") +
                "k" + std::to_string(kv.first % 100u) + "x" +
                std::to_string(kv.second);
        }
        std::fprintf(stderr, "[nnhaul-gates] total=%llu gather_t%u "
                             "publish_t%u%s%s\n",
            static_cast<unsigned long long>(R.executor->gates_total()),
            R.host_act_id[1], R.host_act_id[2], rows.c_str(),
            consumers.c_str());
    }
    if (R.body_breakdown_.load(std::memory_order_relaxed))
    {
        // The per-codelet body histogram, busiest first (the breakdown
        // knob's artifact — the issue loop's work block attributed to
        // the codelet families).
        std::vector<std::pair<std::uint64_t, std::string const *>> rows;
        for (auto const &kv : R.body_ns)
        {
            rows.emplace_back(kv.second.first, &kv.first);
        }
        // Busiest first, NAME-ASCENDING ON TIES — the order is a pure
        // function of the data (the table is unordered now), so the
        // knob's artifact stays run-to-run deterministic.
        std::sort(rows.begin(), rows.end(),
            [](auto const &a, auto const &b) {
                if (a.first != b.first)
                {
                    return a.first > b.first;
                }
                return *a.second < *b.second;
            });
        for (std::size_t i = 0; i < rows.size() && i < 12; ++i)
        {
            std::string const &name = *rows[i].second;
            std::uint64_t const n = R.body_ns.find(name)->second.second;
            std::fprintf(stderr,
                "[nnhaul-body] %s total_ms=%.1f launches=%llu "
                "us_per=%.1f\n",
                name.c_str(),
                static_cast<double>(rows[i].first) / 1.0e6,
                static_cast<unsigned long long>(n),
                n != 0 ? static_cast<double>(rows[i].first) /
                             static_cast<double>(n) / 1.0e3
                       : 0.0);
        }
    }
}

void engine_shutdown()
{
    RuntimeState &R = *rt;

    // NNHAUL_SYMBOL_TRACE (diagnosis only): the full symbol table with
    // per-symbol instance counts, sorted by symbol id — the workload's
    // (op, cl_args, shapes, count) census at teardown.
    if (std::getenv("NNHAUL_SYMBOL_TRACE") != nullptr)
    {
        static const char access_char[3] = {'r', 'w', 'b'};
        std::vector<std::pair<std::uint32_t, SymbolKey const *>> table;
        table.reserve(R.symbol_of.size());
        for (auto const &kv : R.symbol_of)
        {
            table.emplace_back(kv.second, &kv.first);
        }
        std::sort(table.begin(), table.end(),
            [](auto const &a, auto const &b) { return a.first < b.first; });
        for (auto const &entry : table)
        {
            TaskId const id = entry.first;
            SymbolKey const &key = *entry.second;
            std::string hex;
            hex.reserve(key.cl.size() * 2);
            for (unsigned char c : key.cl)
            {
                char buf[4];
                std::snprintf(buf, sizeof(buf), "%02x", c);
                hex += buf;
            }
            auto const name = R.codelet_names.find(key.codelet);
            std::uint64_t count = 0;
            if (auto it = g_symbol_count.find(
                    static_cast<std::uint32_t>(id));
                it != g_symbol_count.end())
            {
                count = it->second;
            }
            std::fprintf(stderr, "[nnhaul-symbols] id=%u count=%llu "
                         "codelet=%s cl=%s bufs=",
                static_cast<unsigned>(id),
                static_cast<unsigned long long>(count),
                name != R.codelet_names.end()
                    ? name->second.c_str()
                    : "?",
                hex.c_str());
            for (BufShape const &s : key.shapes)
            {
                std::fprintf(stderr, "%zu:%zu:%c ", s.n, s.nbytes,
                    access_char[static_cast<int>(s.access)]);
            }
            std::fprintf(stderr, "\n");
        }
    }

    // The final join and drain: join every un-retired batch, close the
    // pending chain, join again (an error is sticky — teardown
    // proceeds regardless). A latched fault may have left a parked host
    // gate behind an already-failed device lane; release it now, or the
    // teardown closes would wait forever.
    if (R.error_set && R.host_lane != nullptr)
    {
        R.host_lane->fault("teardown after a latched plan fault");
    }
    try
    {
        engine_join_all(R, /*force=*/true);
        try
        {
            (void) engine_issue_batch(R);
            engine_join_all(R, /*force=*/true);
        }
        catch (std::exception const &e2)
        {
            R.set_error(e2.what());
        }
    }
    catch (std::exception const &e)
    {
        R.set_error(e.what());
    }
    try
    {
        // Handles still registered die with the runtime: their
        // bindings drop through one last retirement plan.
        std::vector<HandleImpl *> remaining;
        for (auto const &kv : R.live_tiles)
        {
            HandleImpl *impl = kv.second;
            impl->dead.store(true, std::memory_order_relaxed);
            if (impl->dead_prev == nullptr
                && R.dead_pending_head != impl)
            {
                R.dead_pending_push(impl);
            }
        }
        std::vector<PlanOp> ops;
        std::vector<HandleImpl *> swept;
        engine_retire_ops(R, R.seed, ops, swept);
        if (!ops.empty())
        {
            Plan plan;
            plan.plan_seq = R.next_plan_seq;
            plan.entry = R.seed;
            plan.queue = std::move(ops);
            std::vector<SeqTask> seq;
            run_plan(R, plan, seq);
        }
        else
        {
            for (HandleImpl *impl : swept)
            {
                engine_release_swept(R, impl);
            }
        }
    }
    catch (std::exception const &e)
    {
        R.set_error(e.what());
    }
    // Handles outlive this runtime. Their intrusive dead-list links name
    // RuntimeState storage, so clear them before it goes: a later init()
    // re-registers the same handles, and a stale dead_prev would be
    // written through on the next unlink.
    for (auto const &kv : R.live_tiles)
    {
        HandleImpl *impl = kv.second;
        impl->dead_prev = nullptr;
        impl->dead_next = nullptr;
        impl->dead_done = false;
        impl->swept_pending = false;
        impl->pins = 0; // no Acquired from this runtime can still act
        impl->dead.store(false, std::memory_order_relaxed);
    }

    // The trace dump (past the barriers, nothing can append), then
    // disarm — the queue-close barriers retired every hook. The
    // NNHAUL_ALLOC_TRACE counters echo lands here: after every join,
    // before the executor (its issue witnesses) and any resource go.
    (void) R.tracer->dump();
    R.tracer->disarm();
    echo_counters_at_shutdown(R);

    // The explicit release sequence (the
    // member declaration order serves INIT's exception unwinding, not
    // teardown): the barriers ran above; now the host worker stops,
    // then the event pools (first among the resources — deferred
    // destruction, only after the barriers), then the
    // streams (the transfer channels, then the execution lanes — a
    // CudaExecLane's dtor frees its arena beside its stream, the
    // landed backend's own fixed internal order), then the library
    // handles and the fixed device workspaces, then the host pool.
    // Contexts: the engine does NOT cudaDeviceReset — embedding
    // runtimes share the primary contexts and own allocations the
    // engine must not release; context release stays the process
    // exit's act (the previous engine's behavior, stated here).
    // The host lane drains its FIFO on destruction. Its queued bodies
    // call executor-captured lambdas, read the executor's slot operand
    // arrays, record through the host_mint_stream and signal gates
    // through the deferral bridge: all of those must still be alive, so
    // the lane stops FIRST, then its dependencies go.
    R.host_lane->drain_for_teardown(); // FIFO empty; lane still alive
    R.bridge.reset(); // waiters stop (the lane it kicks still exists)
    if (R.host_mint_stream != nullptr)
    {
        // The twins' record stream: after the bridge's waiters stopped
        // and before the provisions that hold the twin events.
        (void) cudaSetDevice(R.ordinals.at(0));
        (void) cudaStreamDestroy(R.host_mint_stream);
        R.host_mint_stream = nullptr;
    }
    R.host_lane.reset(); // now the empty lane stops and joins
    R.executor.reset(); // no queued item names its slots
    R.tracer.reset();
    for (auto &pool : R.event_pools)
    {
        pool.reset(); // the event provisions
    }
    R.channels.clear(); // the transfer streams
    R.exec_lanes.clear(); // the execution streams (arenas beside them)
    for (std::size_t d = 0; d < R.dev.size(); ++d)
    {
        DeviceFacts &f = R.dev[d];
        (void) cudaSetDevice(f.ordinal);
        if (f.cublas != nullptr)
        {
            (void) cublasSetStream(f.cublas, nullptr);
            cublasDestroy(f.cublas);
            f.cublas = nullptr;
        }
        if (f.cublas_workspace != nullptr)
        {
            (void) cudaFree(f.cublas_workspace);
            f.cublas_workspace = nullptr;
        }
        if (f.workspace != nullptr)
        {
            (void) cudaFree(f.workspace);
            f.workspace = nullptr;
        }
        if (f.copy_desc != nullptr)
        {
            (void) cudaFree(f.copy_desc);
            f.copy_desc = nullptr;
        }
    }
    R.host_pool.reset(); // the pinned carve
}

// ---- close-and-plan  ----

void engine_retire_ops(RuntimeState &R,
    std::vector<EntryBinding> const &bindings, std::vector<PlanOp> &ops,
    std::vector<HandleImpl *> &swept)
{
    // A dead tile has no remaining reader: its chunks unlink bare (the
    // planner's own no-remaining-reader case). A pinned handle's run
    // outlives the pin — postpone to the close after the release. The
    // bindings source is the world the unlinks ride ON: the predicted
    // EXIT when they follow a batch's ops (a batch may have moved a
    // dead tile's replicas), the seed for a retire-only plan. The
    // batch walk's own dying frees have normally emptied the dead set
    // of bindings already; this sweep is the belt, indexed per tile so
    // it stays linear in the changed tiles however large the dead set.
    std::unordered_map<TileId, std::vector<EntryBinding const *>> by_tile;
    for (EntryBinding const &b : bindings)
    {
        by_tile[b.tile].push_back(&b);
    }
    for (HandleImpl *impl = R.dead_pending_head; impl != nullptr;)
    {
        HandleImpl *const next = impl->dead_next;
        if (impl->dead_done || impl->swept_pending || impl->pins > 0)
        {
            impl = next;
            continue;
        }
        auto const it = by_tile.find(impl->tile_id);
        if (it != by_tile.end())
        {
            for (EntryBinding const *b : it->second)
            {
                ops.push_back(
                    make_unlink(b->tile, b->index, b->node, b->chunk));
            }
        }
        swept.push_back(impl);
        impl = next;
    }
}

GraphEnd engine_issue_batch(RuntimeState &R)
{
    std::lock_guard<std::mutex> lock(R.mu);
    if (R.error_set)
    {
        throw Error(R.error);
    }

    // the run-act head brackets, declared early: take_pending and
    // verify_pins run before the main phase-variable block below (the
    // empty-close early return sits between). Diagnosis only, never
    // control flow — the same NNHAUL_BOUNDARY_BREAKDOWN knob.
    bool const boundary_head = boundary_breakdown_on();
    // Take the pending chain first (submit order).
    Task *first = nullptr;
    Task *last = nullptr;
    {
        std::lock_guard<std::mutex> sm(R.submit_mu);
        R.queue.take_pending(first, last);
    }
    std::uint64_t const tb_take = boundary_head ? now_ns() : 0;

    // The pinned-run safety net, before anything plans over the seed.
    if (first != nullptr || !R.dead_pending_empty())
    {
        verify_pins(R);
    }
    std::uint64_t const tb_pins = boundary_head ? now_ns() : 0;

    if (first == nullptr)
    {
        // An empty close: no batch — only the retirement, as its own
        // retire-only plan over the current seed, and only when
        // nothing stands un-retired (the old engine's Idle discipline:
        // a retirement needs a joined world). With batches in flight
        // the close is dropped — the join will carry it.
        if (!R.unretired.empty())
        {
            return GraphEnd{0, true};
        }
        std::vector<PlanOp> ops;
        std::vector<HandleImpl *> swept;
        engine_retire_ops(R, R.seed, ops, swept);
        if (!ops.empty())
        {
            Plan plan;
            plan.plan_seq = R.next_plan_seq;
            plan.entry = R.seed;
            plan.queue = std::move(ops);
            std::vector<SeqTask> seq;
            run_plan(R, plan, seq);
        }
        else
        {
            for (HandleImpl *impl : swept)
            {
                engine_release_swept(R, impl);
            }
        }
        return GraphEnd{0, false};
    }

    // Depth-2: at most two un-retired batches. A third
    // at a non-empty close is the LOUD pipeline-depth abort (the old
    // engine's step 3, restored) — the caller failed to join.
    if (R.unretired.size() >= 2)
    {
        std::fprintf(stderr,
            "nnhaul: pipeline depth exceeded: two un-retired batches "
            "and no join\n");
        std::string const msg =
            "pipeline depth exceeded: call join_predecessor() before "
            "the next non-empty close";
        R.set_error(msg);
        for (Task *t = first; t != nullptr;)
        {
            Task *next = t->next;
            R.arena.free(t);
            t = next;
        }
        throw Error(msg);
    }

    // The batch's request: TaskId = the task's SYMBOL id (many
    // instances of one
    // (codelet, cl_args, shapes) triple share the symbol and its one
    // registration; the collapse rule carries the repeats),
    // the node recorded at submit, the operands from the buffers'
    // tile ids and accesses. The dead-tile set is the riding
    // retirement — the planner's cross-batch survival input (a dead
    // tile's bytes may die at an eviction; a LIVE tile's last actual
    // replica is rescued always: its readers may live in later
    // batches — the seam, closed engine-side).
    bool const breakdown =
        std::getenv("NNHAUL_ISSUE_BREAKDOWN") != nullptr;
    bool const boundary = boundary_breakdown_on();
    std::uint64_t t_reg = 0, t_plan = 0, t_check = 0, t_issue = 0;
    std::uint64_t const t0 = breakdown ? now_ns() : 0;
    // The boundary brackets (diagnosis only, never control flow):
    // absolute steady-clock nanoseconds at this close's phase walls —
    // merged offline with the executor's [nnhaul-boundary-exe] and
    // [nnhaul-boundary-close] lines and the trainer's per-iter phase
    // lines, they name what the wait-phase drain tail parks on.
    std::uint64_t const tb_enter = boundary ? now_ns() : 0;
    // the run-act head split (diagnosis only, never control flow):
    // take (the pending-chain drain) / pins (verify_pins) / reg (the
    // FUSED register+request walk) — with plan0 they tile the
    // engine_issue_batch head the old line read as one block. tb_take/
    // tb_pins were read before the empty-close return; on the steady
    // (non-empty) path they precede tb_enter by microseconds.
    std::uint64_t tb_reg = 0, tb_comp1 = 0, tb_log = 0;
    std::uint64_t tb_plan0 = 0, tb_plan1 = 0, tb_check1 = 0,
                  tb_issue0 = 0;
    PlanRequest req;
    req.plan_seq = R.next_plan_seq;
    req.entry = R.seed;
    for (HandleImpl *impl = R.dead_pending_head; impl != nullptr;
         impl = impl->dead_next)
    {
        if (!impl->dead_done && impl->pins == 0)
        {
            req.dead_tiles.push_back(impl->tile_id);
        }
    }

    // The register+request pass, FUSED over ONE walk of the
    // pending chain: the giant single-shot plan's 47,781-task chain is
    // a scattered-heap pointer chase, and the old two walks (intern,
    // then the PlannedTask build) doubled it — the registration phase
    // was a large per-task cost at giant plans. One walk, both
    // products, same order, same symbols (engine_register_batch's
    // body inlined per task). t_reg is its ONE wall (the fused walk
    // admits no honest reg/req split — the pre-fusion t_req bracket
    // read a second clock a few ns after t_reg and carried no phase
    // of its own).
    {
        for (Task *t = first; t != nullptr; t = t->next)
        {
            std::vector<BufShape> shapes;
            shapes.reserve(t->nbufs);
            PlannedTask pt;
            pt.node = t->node;
            pt.operands.reserve(t->nbufs);
            std::set<TileId> seen_tiles;
            for (std::size_t i = 0; i < t->nbufs; ++i)
            {
                Task::Buf const &buf = t->bufs()[i];
                HandleImpl const &impl = *buf.handle->impl_;
                // The launch passes one pointer run per BUFFER, in submit
                // order. Two buffers naming one tile cannot be expressed
                // by a TILE_OP's distinct (tile, chunk) pairs, so refuse
                // the task instead of issuing a short operand array.
                if (!seen_tiles.insert(impl.tile_id).second)
                {
                    std::string const msg = "task "
                        + std::to_string(req.tasks.size()) + " names tile "
                        + std::to_string(impl.tile_id) + " in two "
                          "buffers; a task's operands must name distinct "
                          "tiles";
                    R.set_error(msg);
                    throw Error(msg);
                }
                shapes.push_back(
                    BufShape{impl.n, impl.nbytes, buf.access});
                pt.operands.push_back(
                    TileUse{impl.tile_id, buf.access});
            }
            TaskId const symbol = engine_intern_symbol(
                R, *t->codelet, t->cl_data(), t->cl_size, shapes);
            engine_ensure_symbol_registered(
                R, symbol, t->node, *t->codelet, t->cl_data(),
                t->cl_size, shapes);
            ++g_symbol_count[static_cast<std::uint32_t>(symbol)];
            pt.task = symbol;
            req.tasks.push_back(std::move(pt));
        }
        tb_reg = boundary ? now_ns() : 0;
    }

    // Plan (the lowering walk over the state read).
    t_reg = breakdown ? now_ns() - t0 : 0;
    std::uint64_t const t1 = breakdown ? now_ns() : 0;
    tb_plan0 = boundary ? now_ns() : 0;
    PlannedBatch pb;
    try
    {
        pb = planner::run(R.planner_plugin, req, *R.inv, R.tiles,
            R.channel_rates);
    }
    catch (PlanCensusError const &e)
    {
        std::string msg =
            std::string("plan-time census: ") + census_text(e.census);
        std::fprintf(stderr, "nnhaul: %s\n", msg.c_str());
        R.set_error(msg);
        throw Error(msg);
    }
    catch (std::exception const &e)
    {
        R.set_error(e.what());
        throw Error(e.what());
    }
    catch (...)
    {
        // plan() is plugin code: a non-standard throw must still latch
        // the runtime, or a later batch would run on a stale world view.
        // name() is plugin code too: it must not escape this handler.
        char const *planner_name = "nnhaul.lru";
        if (R.planner_plugin != nullptr)
        {
            try
            {
                planner_name = R.planner_plugin->name();
            }
            catch (...)
            {
                planner_name = "<plugin>";
            }
            if (planner_name == nullptr)
            {
                planner_name = "<plugin>";
            }
        }
        std::string const msg = std::string("planner '") + planner_name
            + "' threw a non-standard exception";
        R.set_error(msg);
        throw Error(msg);
    }

    // The churn meters (the per-plan line): the
    // walk's counts plus the dead residue the entry carried — the
    // residency story of the batch in one line, under the counters
    // echo's own knob. With NNHAUL_ISSUE_BREAKDOWN the phase walls
    // ride along (register+request, plan, check, issue — the old
    // engine's diagnostic knob's discipline: env-gated clock reads,
    // off by default, never control flow).
    t_plan = breakdown ? now_ns() - t1 : 0;
    tb_plan1 = boundary ? now_ns() : 0;
    std::uint64_t const t2 = breakdown ? now_ns() : 0;
    R.evictions += pb.stats.evictions;
    R.rescues += pb.stats.rescues;
    R.refetches += pb.stats.refetches;
    R.prefetches += pb.stats.prefetches;
    R.dying_frees += pb.stats.dying_frees;
    if (std::getenv("NNHAUL_ALLOC_TRACE") != nullptr
        || std::getenv("NNHAUL_PLAN_DUMP") != nullptr)
    {
        std::unordered_set<TileId> const dead(
            req.dead_tiles.begin(), req.dead_tiles.end());
        std::size_t dead_in_entry = 0;
        std::size_t host_ops = 0;
        // The plan dump (diagnosis only): NNHAUL_PLAN_DUMP=a,b,c prints
        // every op of the named plan sequences, one op per line — the
        // placement analysis (eviction/refetch positions against the
        // TILE_OP stream) reads this offline. Never control flow.
        char const *dump_env = std::getenv("NNHAUL_PLAN_DUMP");
        if (dump_env != nullptr)
        {
            std::vector<std::uint64_t> want;
            for (char const *p = dump_env; *p != '\0';)
            {
                char *end = nullptr;
                std::uint64_t const v = std::strtoull(p, &end, 10);
                if (end == p)
                {
                    break;
                }
                want.push_back(v);
                p = end;
                while (*p == ',' || *p == ' ')
                {
                    ++p;
                }
            }
            if (std::find(want.begin(), want.end(),
                    static_cast<std::uint64_t>(req.plan_seq)) !=
                want.end())
            {
                std::fprintf(stderr, "[nnhaul-dump] seq=%llu begin\n",
                    static_cast<unsigned long long>(req.plan_seq));
                std::size_t task_no = 0;
                for (std::size_t q = 0; q < pb.plan.queue.size(); ++q)
                {
                    PlanOp const &op = pb.plan.queue[q];
                    switch (op.kind)
                    {
                    case PlanOpKind::Link:
                        std::fprintf(stderr,
                            "[nnhaul-dump] %zu L t=%u i=%u n=%u c=%u "
                            "nb=%zu\n",
                            q, op.u.link.tile, op.u.link.index,
                            op.u.link.node, op.u.link.chunk,
                            R.tiles.nbytes(op.u.link.tile));
                        break;
                    case PlanOpKind::Unlink:
                        std::fprintf(stderr,
                            "[nnhaul-dump] %zu U t=%u i=%u n=%u c=%u "
                            "nb=%zu\n",
                            q, op.u.unlink.tile, op.u.unlink.index,
                            op.u.unlink.node, op.u.unlink.chunk,
                            R.tiles.nbytes(op.u.unlink.tile));
                        break;
                    case PlanOpKind::Copy:
                        std::fprintf(stderr,
                            "[nnhaul-dump] %zu C t=%u i=%u "
                            "s=%u.%u d=%u.%u nb=%zu\n",
                            q, op.u.copy.tile, op.u.copy.index,
                            op.u.copy.src.node, op.u.copy.src.chunk,
                            op.u.copy.dst.node, op.u.copy.dst.chunk,
                            R.tiles.nbytes(op.u.copy.tile));
                        break;
                    case PlanOpKind::TileOp:
                    {
                        std::fprintf(stderr,
                            "[nnhaul-dump] %zu T k=%llu n=%u ops=%zu",
                            q,
                            static_cast<unsigned long long>(
                                op.u.tile_op.task),
                            op.u.tile_op.node, op.operands.size());
                        for (Operand const &o : op.operands)
                        {
                            std::fprintf(stderr, " %u.%u.%u.%c",
                                o.tile, o.index, o.chunk,
                                o.access == Access::Read
                                    ? 'r'
                                    : (o.access == Access::Write
                                             ? 'w'
                                             : 'b'));
                        }
                        std::fprintf(stderr, "\n");
                        ++task_no;
                        break;
                    }
                    }
                }
                std::fprintf(stderr, "[nnhaul-dump] seq=%llu end\n",
                    static_cast<unsigned long long>(req.plan_seq));
            }
        }
        for (EntryBinding const &b : req.entry)
        {
            if (dead.count(b.tile) != 0)
            {
                dead_in_entry += 1;
            }
        }
        for (PlanOp const &op : pb.plan.queue)
        {
            if (op.kind == PlanOpKind::TileOp &&
                R.inv->is_host(op.u.tile_op.node))
            {
                host_ops += 1;
            }
        }
        if (std::getenv("NNHAUL_ALLOC_TRACE") != nullptr)
        {
        std::fprintf(stderr,
            "[nnhaul-plan] seq=%llu tasks=%zu ops=%zu entry=%zu "
            "dead_entry=%zu exit=%zu host_ops=%zu evict=%llu "
            "rescue=%llu refetch=%llu prefetch=%llu dyingfree=%llu "
            "runmiss=%llu want=%lu longest=%lu spend=%llu\n",
            static_cast<unsigned long long>(req.plan_seq),
            req.tasks.size(), pb.plan.queue.size(), req.entry.size(),
            dead_in_entry, pb.exit.size(), host_ops,
            static_cast<unsigned long long>(pb.stats.evictions),
            static_cast<unsigned long long>(pb.stats.rescues),
            static_cast<unsigned long long>(pb.stats.refetches),
            static_cast<unsigned long long>(pb.stats.prefetches),
            static_cast<unsigned long long>(pb.stats.dying_frees),
            static_cast<unsigned long long>(pb.stats.run_misses),
            static_cast<std::size_t>(pb.stats.run_miss_want),
            pb.stats.run_miss_longest,
            static_cast<unsigned long long>(pb.stats.pin_spends));
        }
    }

    // Compose: the planner's product, then the retirement TRAILING it
    // — the belt over the planner's own DYING-VICTIM FREE (the
    // planner's dying-free order): the walk already unlinked every dead tile's bindings
    // at their dying positions, so this sweep normally finds nothing,
    // but it stays the engine's guarantee that a dead tile leaves the
    // world even for the shapes the walk cannot free early (a dead
    // tile's readers live in THIS batch — invalidate orders the drop
    // after every task already submitted), and its unlinks name the
    // bindings of the planner's PREDICTED EXIT (the batch may have
    // moved them), never the seed's.
    Plan plan = std::move(pb.plan);

    // Check (FULL inputs, the chain ALWAYS) — no plan reaches issue
    // without a passed check.
    apply_plan_growth(R, plan); // grow before the check
    std::vector<SeqTask> seq = to_seq_tasks(req.tasks);
    CheckInput in;
    in.plan = &plan;
    in.inventory = &*R.inv;
    in.tiles = &R.tiles;
    in.runnable = &R.runnable;
    in.task_sequence = &seq;
    in.chain = R.seed;
    tb_comp1 = boundary ? now_ns() : 0;
    CheckResult verdict = check(in);
    if (!verdict.pass)
    {
        std::string msg =
            verdict.census.has_value()
                ? "check-time census: " + census_text(*verdict.census)
                : "check-time census: " + verdict.fault;
        std::fprintf(stderr, "nnhaul: %s\n", msg.c_str());
        R.set_error(msg);
        throw Error(msg);
    }

    // ISSUE : the executor enqueues the whole queue and
    // returns — its barriers do NOT run here; the batch joins at
    // join_predecessor()/wait(). The seed advances to the checked exit
    // NOW (the successor seeds and issues while this batch's lanes
    // still run — enqueue-before-derive is the ENQUEUE order,
    // which the executor's discipline assertion holds by
    // construction).
    t_check = breakdown ? now_ns() - t2 : 0;
    tb_check1 = boundary ? now_ns() : 0;
    std::uint64_t const t3 = breakdown ? now_ns() : 0;
    tb_issue0 = boundary ? now_ns() : 0;
    // One launch per task occurrence with its FULL operand set, in the
    // submission's buffer order and with each tile's chunks ascending.
    // The checker validates the union over consecutive TILE_OPs (a split
    // is legal there), but the executor hands each TILE_OP's array
    // straight to the codelet, so a short, permuted or extra entry would
    // be read out of bounds or bind the wrong pointers to a buffer.
    // Enforce the per-launch shape before anything is issued.
    {
        auto fail = [&R](std::string const &msg) {
            R.set_error(msg);
            throw Error(msg);
        };
        std::size_t occ = 0;
        for (PlanOp const &op : plan.queue)
        {
            if (op.kind != PlanOpKind::TileOp)
            {
                continue;
            }
            if (occ >= req.tasks.size()
                || req.tasks[occ].task != op.u.tile_op.task)
            {
                fail("plan shape: TILE_OP for task "
                    + std::to_string(op.u.tile_op.task)
                    + " is not the next task occurrence");
            }
            if (op.u.tile_op.node != req.tasks[occ].node)
            {
                fail("plan shape: task "
                    + std::to_string(op.u.tile_op.task) + " is placed on "
                    + std::to_string(op.u.tile_op.node)
                    + ", but was submitted on node "
                    + std::to_string(req.tasks[occ].node));
            }
            std::size_t oi = 0;
            for (TileUse const &use : req.tasks[occ].operands)
            {
                ChunkIndex const n = R.tiles.n(use.tile);
                for (ChunkIndex i = 0; i < n; ++i)
                {
                    if (oi >= op.operands.size())
                    {
                        fail("plan shape: task "
                            + std::to_string(op.u.tile_op.task)
                            + " has too few operands in one TILE_OP");
                    }
                    Operand const &got = op.operands[oi];
                    if (got.tile != use.tile || got.index != i)
                    {
                        fail("plan shape: task "
                            + std::to_string(op.u.tile_op.task) + " needs "
                            "operands in submission order (tile "
                            + std::to_string(use.tile) + ", chunk "
                            + std::to_string(i) + ") at position "
                            + std::to_string(oi) + ", got (tile "
                            + std::to_string(got.tile) + ", chunk "
                            + std::to_string(got.index) + ")");
                    }
                    ++oi;
                }
            }
            if (oi != op.operands.size())
            {
                fail("plan shape: task "
                    + std::to_string(op.u.tile_op.task) + " has "
                    + std::to_string(op.operands.size() - oi)
                    + " operand(s) that no submitted buffer names");
            }
            ++occ;
        }
        if (occ != req.tasks.size())
        {
            fail("plan shape: " + std::to_string(req.tasks.size() - occ)
                + " task(s) without a TILE_OP");
        }
    }

    // Retire dead tiles from the CHECKER's exit fold, not the planner's
    // prediction: the fold is the state the plan actually produced, so
    // the appended unlinks name bindings that exist. A planner exit that
    // omits or misbinds a dead binding cannot leak chunks this way. The
    // unlinks join the queue BEFORE issue: close() reconciles the issued
    // log against plan.queue, so a trailing append would never be issued.
    std::vector<HandleImpl *> swept;
    {
        std::vector<PlanOp> retire_ops;
        engine_retire_ops(R, verdict.exit, retire_ops, swept);
        if (!retire_ops.empty())
        {
            plan.queue.insert(plan.queue.end(), retire_ops.begin(),
                retire_ops.end());
            // The seed is the fold MINUS what just retired.
            auto &ex = verdict.exit;
            ex.erase(std::remove_if(ex.begin(), ex.end(),
                         [&](EntryBinding const &b) {
                             for (PlanOp const &op : retire_ops)
                             {
                                 if (op.kind == PlanOpKind::Unlink
                                     && op.u.unlink.node == b.node
                                     && op.u.unlink.chunk == b.chunk)
                                 {
                                     return true;
                                 }
                             }
                             return false;
                         }),
                ex.end());
        }
    }

    std::size_t slot = 0;
    try
    {
        slot = R.executor->issue(plan);
    }
    catch (ExecutorCensusError const &e)
    {
        std::string msg = std::string("execution census: ") + e.what();
        if (!e.census.inputs.empty())
        {
            msg += "\n";
            msg += e.census.inputs;
        }
        R.set_error(msg);
        throw Error(msg);
    }
    catch (std::exception const &e)
    {
        R.set_error(e.what());
        throw Error(e.what());
    }
    // The retirement ops are issued and owned by this un-retired batch:
    // a following close must not sweep the same impl again before this
    // batch joins. Cleared by engine_release_swept at the join.
    for (HandleImpl *impl : swept)
    {
        impl->swept_pending = true;
    }
    R.seed = verdict.exit;
    R.next_plan_seq = plan.plan_seq + 1;
    R.model_read_only = true;
    R.batch_counter += 1;
    R.batches_planned += 1;
    R.planned_ops += plan.queue.size();
    tb_log = boundary ? now_ns() : 0;
    // t_exit read HERE, before the [nnhaul-issue] print: with both
    // knobs on, the print's own fprintf must not ride inside the
    // t_log -> t_exit window (the bracket closes at the issue act's
    // last engine work, not after a diagnostic line).
    std::uint64_t const tb_exit = boundary ? now_ns() : 0;
    if (breakdown)
    {
        t_issue = now_ns() - t3;
        std::fprintf(stderr,
            "[nnhaul-issue] seq=%llu reg_ms=%.2f plan_ms=%.2f "
            "check_ms=%.2f issue_ms=%.2f\n",
            static_cast<unsigned long long>(req.plan_seq),
            static_cast<double>(t_reg) / 1.0e6,
            static_cast<double>(t_plan) / 1.0e6,
            static_cast<double>(t_check) / 1.0e6,
            static_cast<double>(t_issue) / 1.0e6);
    }
    if (boundary)
    {
        std::fprintf(stderr,
            "[nnhaul-boundary-plan] seq=%llu t_enter=%llu "
            "t_take=%llu t_pins=%llu t_reg=%llu "
            "t_plan0=%llu t_plan1=%llu t_comp1=%llu t_check1=%llu "
            "t_issue0=%llu t_log=%llu t_exit=%llu\n",
            static_cast<unsigned long long>(req.plan_seq),
            static_cast<unsigned long long>(tb_enter),
            static_cast<unsigned long long>(tb_take),
            static_cast<unsigned long long>(tb_pins),
            static_cast<unsigned long long>(tb_reg),
            static_cast<unsigned long long>(tb_plan0),
            static_cast<unsigned long long>(tb_plan1),
            static_cast<unsigned long long>(tb_comp1),
            static_cast<unsigned long long>(tb_check1),
            static_cast<unsigned long long>(tb_issue0),
            static_cast<unsigned long long>(tb_log),
            static_cast<unsigned long long>(tb_exit));
    }
    // The task/copy history is a bounded diagnostic window, not a
    // permanent per-op growth log. Only un-retired batches (at most one
    // here, before this batch is queued) carry offsets into task_log.
    {
        constexpr std::size_t kLogCap = std::size_t{1} << 18;
        if (R.task_log.size() > kLogCap)
        {
            std::size_t const drop = R.task_log.size() - kLogCap;
            R.task_log.erase(R.task_log.begin(),
                R.task_log.begin() + static_cast<std::ptrdiff_t>(drop));
            for (RuntimeState::IssuedBatch &b : R.unretired)
            {
                b.task_log_from = b.task_log_from > drop
                    ? b.task_log_from - drop : 0;
            }
        }
        if (R.copy_log_.size() > kLogCap)
        {
            R.copy_log_.erase(R.copy_log_.begin(),
                R.copy_log_.end()
                    - static_cast<std::ptrdiff_t>(kLogCap));
        }
    }
    // views enter the log as RUNNING (their plan issued; the op not
    // yet joined) and flip to Done at the join.
    RuntimeState::IssuedBatch record;
    record.batch = R.batch_counter;
    record.plan_seq = plan.plan_seq;
    record.slot = slot;
    record.task_log_from = R.task_log.size();
    for (Task *t = first; t != nullptr; t = t->next)
    {
        TaskView v;
        v.submit_seq = t->submit_seq;
        v.worker = static_cast<WorkerId>(t->node);
        v.kind = R.inv->is_host(t->node) ? Kind::Cpu : Kind::Cuda;
        v.footprint = t->footprint;
        v.phase = TaskPhase::Running;
        v.codelet = t->codelet != nullptr ? t->codelet->name()
                                          : std::string();
        R.task_log.push_back(std::move(v));
    }
    record.tasks = first;
    record.swept = std::move(swept);
    record.deriv = R.executor->slot_derivation(slot);
    record.plan = std::move(plan);
    R.unretired.push_back(std::move(record));

    // The 1-based batch number (0 stays "nothing" — run_pending
    // disambiguates, the GraphEnd result).
    return GraphEnd{R.batch_counter, true};
}

std::uint64_t engine_join_oldest(RuntimeState &R)
{
    std::lock_guard<std::mutex> lock(R.mu);
    if (R.unretired.empty())
    {
        return 0;
    }
    if (R.error_set)
    {
        throw Error(R.error);
    }
    // Keep the record in the queue until close succeeds: a close fault
    // must leave the batch for the forced teardown path, not drop its
    // slot, tasks and swept handles on the floor.
    RuntimeState::IssuedBatch &front_record = R.unretired.front();

    // The boundary brackets on the join act (diagnosis only): the
    // trainer's iteration-end wait parks inside executor->close — the
    // absolute walls here bracket the park (t_enter -> t_close1) and
    // the engine-side post-join host work (t_close1 -> t_exit) that
    // delays the successor iteration's record/compile/issue.
    bool const boundary = boundary_breakdown_on();
    std::uint64_t const tb_enter = boundary ? now_ns() : 0;
    std::uint64_t tb_close0 = 0, tb_close1 = 0;

    // The JOIN act: the record's queue-close barriers (reconcile,
    // barriers over ALL lanes, registry sweep, tracer harvest), then
    // the engine's retirement accounting. (The witness is read
    // through last_issue() — close() no longer returns a deep copy;
    // the accounting below walks the same issue_log it always did.)
    try
    {
        tb_close0 = boundary ? now_ns() : 0;
        R.executor->close(front_record.plan, front_record.slot);
        tb_close1 = boundary ? now_ns() : 0;
    }
    catch (ExecutorCensusError const &e)
    {
        std::string msg = std::string("execution census: ") + e.what();
        if (!e.census.inputs.empty())
        {
            msg += "\n";
            msg += e.census.inputs;
        }
        R.set_error(msg);
        throw Error(msg);
    }
    catch (std::exception const &e)
    {
        R.set_error(e.what());
        throw Error(e.what());
    }

    // The close succeeded: take the record out now, before accounting
    // that could throw. A record retained past a successful close would
    // be closed a second time by the forced teardown path.
    RuntimeState::IssuedBatch record = std::move(front_record);
    R.unretired.pop_front();

    engine_note_batch(R, record.plan, R.executor->last_issue());
    engine_simulate_plan(R, record.plan, record.deriv);

    // The batch's task views flip to Done; the arena takes the nodes.
    std::size_t ntasks = 0;
    for (Task *t = record.tasks; t != nullptr; t = t->next)
    {
        ++ntasks;
    }
    for (std::size_t i = record.task_log_from;
         i < R.task_log.size() && i < record.task_log_from + ntasks;
         ++i)
    {
        R.task_log[i].phase = TaskPhase::Done;
    }
    for (Task *t = record.tasks; t != nullptr;)
    {
        Task *next = t->next;
        R.arena.free(t);
        t = next;
    }
    for (HandleImpl *impl : record.swept)
    {
        engine_release_swept(R, impl);
    }
    if (boundary)
    {
        std::fprintf(stderr,
            "[nnhaul-boundary-join] seq=%llu t_enter=%llu "
            "t_close0=%llu t_close1=%llu t_exit=%llu\n",
            static_cast<unsigned long long>(record.plan.plan_seq),
            static_cast<unsigned long long>(tb_enter),
            static_cast<unsigned long long>(tb_close0),
            static_cast<unsigned long long>(tb_close1),
            static_cast<unsigned long long>(now_ns()));
    }
    return record.batch;
}

void engine_join_all(RuntimeState &R, bool force)
{
    if (force)
    {
        // Teardown discipline: join past a sticky error; each close
        // delivers its own fault census if one stuck.
        for (;;)
        {
            std::lock_guard<std::mutex> lock(R.mu);
            if (R.unretired.empty())
            {
                return;
            }
            RuntimeState::IssuedBatch &record = R.unretired.front();
            try
            {
                (void) R.executor->close(record.plan, record.slot);
            }
            catch (std::exception const &e)
            {
                R.set_error(e.what());
            }
            for (Task *t = record.tasks; t != nullptr;)
            {
                Task *next = t->next;
                R.arena.free(t);
                t = next;
            }
            for (HandleImpl *impl : record.swept)
            {
                engine_release_swept(R, impl);
            }
            for (HandleImpl *impl = R.dead_pending_head; impl != nullptr;)
            {
                HandleImpl *const next = impl->dead_next;
                if (impl->dead_done)
                {
                    R.dead_pending_unlink(impl);
                }
                impl = next;
            }
            R.unretired.pop_front();
        }
    }
    while (!R.unretired.empty())
    {
        (void) engine_join_oldest(R);
    }
}

void engine_sync_close(RuntimeState &R)
{
    // The boundary acts' fused path: a fully joined world, then the
    // pending batch (issued and joined at once), then joined again —
    // wait(), every host acquire, the Handle destructor, shutdown.
    engine_join_all(R);
    (void) engine_issue_batch(R);
    engine_join_all(R);
}

// ---- the host acquires  ----

void *engine_acquire(RuntimeState &R, HandleImpl &impl, Access access)
{
    // NO WORLD JOIN (the boundary acquire shape):
    // the pending batch ISSUES —
    // its tasks may write the acquired tile, so the mini-plan's seed
    // must carry them — joining the OLDEST first only when depth-2
    // stands full (the guard's discipline; the old engine joined the
    // previous batch while the current one was already pushed, briefly,
    // never join-all-then-issue-then-join-all). The mini-plan below
    // then orders ITSELF against the in-flight producers: its D2H
    // copies wait the producers' mints (late records into still-
    // running lanes are the executor's native case), the gather gates
    // host-side (a HOST-lane park, lane-internal), and only the
    // mini-plan's OWN terminal events join — the user's bytes, never
    // the world.
    if (R.unretired.size() >= 2)
    {
        (void) engine_join_oldest(R);
    }
    (void) engine_issue_batch(R);

    std::lock_guard<std::mutex> lock(R.mu);
    if (R.error_set)
    {
        throw Error(R.error);
    }
    if (impl.dead.load(std::memory_order_relaxed))
    {
        throw Error("handle \"" + label_of(impl) + "\" is dead");
    }

    NodeId const host = R.inv->host_id();
    std::size_t const n = impl.n;
    SeedIndex world(R.seed);

    // An existing active run of this handle: pin it again (the old
    // engine allowed stacked acquires; the bytes are the same blocks).
    bool const publish = access != Access::Read;
    auto held = R.acquired.find(&impl);
    if (held != R.acquired.end())
    {
        // A stacked acquire pins the same run. A Write over a
        // Read-created run must mark it for publication: the host is
        // about to change the bytes, and the release has to publish them
        // or later device reads see the stale replica.
        impl.pins += 1;
        if(access != Access::Read)
        {
            held->second.publish = true;
        }
        return R.host_pool->addr(held->second.base);
    }

    // Fast path: the tile's chunks already sit on a contiguous host
    // run — actual — AND the issued world stands RETIRED (every
    // issued plan joined: the seed's actuality is then LANDED bytes,
    // not an in-flight production; the old engine's post-join
    // current_run, at the pipelined executor's granularity). With
    // batches still in flight the slow path takes over: its copies
    // wait the producers and the join rides the mini-plan's own
    // terminals — sound without any world join.
    if (R.unretired.empty())
    {
        auto b0 = world.binding_of(impl.tile_id, ChunkIndex{0}, host);
        if (b0.has_value())
        {
            bool const need_actual = access != Access::Write;
            bool ok = need_actual ? b0->second : true;
            // One pointer is handed out for the whole tile, so the run
            // must also lie inside ONE host allocation. Consecutive
            // ChunkIds can straddle a cudaHostAlloc segment boundary;
            // the slow carve rejects that and this path must too.
            if (ok && R.host_pool->contig_bytes(b0->first)
                    < n * chunk_class_bytes())
            {
                ok = false;
            }
            for (std::size_t i = 1; ok && i < n; ++i)
            {
                auto bi = world.binding_of(impl.tile_id,
                    static_cast<ChunkIndex>(i), host);
                if (!bi.has_value() || bi->first != b0->first + i ||
                    (need_actual && !bi->second))
                {
                    ok = false;
                }
            }
            if (ok)
            {
                impl.pins += 1;
                R.acquired.emplace(&impl,
                    AcquireRun{b0->first, n, publish && !impl.dead.load(
                        std::memory_order_relaxed)});
                return R.host_pool->addr(b0->first);
            }
        }
    }

    // The contiguous-run carve: n consecutive host blocks, each FREE
    // or EVICTABLE (the Acquired surface hands out one pointer for the
    // whole tile, and the pool's carve is contiguous). Eviction
    // is a plan act with the data-survival rule: a victim that is the
    // last ACTUAL replica of its logical chunk is rescued onto a free
    // device chunk before its UNLINK (the planner's own discipline; a
    // boundary act has no batch knowledge, so the bytes survive
    // always). Blocks under a live pin never evict.
    // GROW FIRST: the host acquire needs n contiguous
    // whole-class blocks — grow the pool by exactly the shortfall
    // (bounded by the soft limit) before any window search; only a
    // pool at its limit falls through to the run search over what is
    // held. (A growth here is a boundary act, committed directly —
    // the binding table's host-node segment grows by the same blocks;
    // they are FREE, so no plan boundary ever sees them.)
    {
        std::size_t const card = R.host_pool->block_count();
        if (n > card)
        {
            // The acquire's TILE-SIZED need
            // commits through the act ladder (the tile's own segment
            // first); the window search below works within whatever
            // budget the wall left.
            if (grow_pool_for(R, R.inv->host_id(), n))
            {
                NodeId const host = R.inv->host_id();
                R.inv->grow_node(host, R.host_pool->block_count());
                if (R.executor != nullptr)
                {
                    R.executor->apply_growth(
                        host, R.host_pool->block_count());
                }
                sync_event_provisions(R);
            }
        }
    }
    HostBlocks blocks(R);
    std::unordered_set<ChunkId> pinned;
    for (auto const &kv : R.acquired)
    {
        for (std::size_t i = 0; i < kv.second.n; ++i)
        {
            pinned.insert(
                static_cast<ChunkId>(kv.second.base + i));
        }
    }

    if (n > blocks.count())
    {
        throw Error("tile \"" + label_of(impl) + "\" (" +
                    std::to_string(impl.nbytes) + " bytes, " +
                    std::to_string(n) + " chunks) is too large for the "
                    "host pool of " + std::to_string(blocks.count()) +
                    " blocks — a host acquire needs one contiguous "
                    "whole-class run");
    }

    struct Window
    {
        ChunkId base = 0;
        std::vector<ChunkId> victims; // occupied blocks in the window
    };
    // GROW FIRST: the initial search covers the blocks
    // held NOW; if no window emerges, the pool grows n more blocks
    // (bounded by the soft limit) and the search resumes over ONLY
    // THE FRESH TAIL — never a rebuilt view: the blocks view is
    // unchanged by the growth itself (victims commit only AFTER a
    // window is chosen, below), but it carries the evictions of EARLIER
    // acquires of this same tile-pinning epoch, and rebuilding from
    // R.seed would resurrect those evicted victims' bindings. The
    // tail's fresh run is contiguous, free, and inside one segment by
    // construction, so one successful growth ends it.
    std::optional<Window> window;
    std::size_t searched = 0; // [0, searched) already examined
    for (;;)
    {
        std::size_t const held_now = blocks.count();
        for (std::size_t b = searched;
             b + n <= held_now && !window.has_value(); ++b)
        {
            searched = b + 1;
            Window w;
            w.base = static_cast<ChunkId>(b);
            bool ok = true;
            // One pointer for the whole tile: the run must
            // sit inside ONE host segment (its allocation).
            if (R.host_pool->contig_bytes(b) < n * chunk_class_bytes())
            {
                continue;
            }
            for (std::size_t i = 0; ok && i < n; ++i)
            {
                if (blocks.free_[b + i])
                {
                    continue;
                }
                ChunkId const block = static_cast<ChunkId>(b + i);
                if (pinned.find(block) != pinned.end())
                {
                    ok = false; // a live pin's block never evicts
                    break;
                }
                // The ACQUIRING tile's own binding is no victim — the
                // own-binding pass below owns it (reuse at its run
                // position, unlink elsewhere).
                if (blocks.held_[b + i].has_value() &&
                    blocks.held_[b + i]->tile == impl.tile_id)
                {
                    continue;
                }
                w.victims.push_back(block);
            }
            if (ok)
            {
                window = std::move(w);
            }
        }
        if (window.has_value())
        {
            break;
        }
        // Grow n more blocks (bounded by the soft limit). The full
        // commit keeps the inventory, the executor's applied table,
        // and the dependency rows in step. The fresh tail is free by
        // construction — extend the blocks view in place, never a
        // rebuild.
        // One tile-sized commit for the
        // acquire's need; if it adds nothing (the pool at its wall or
        // closed), the last search stands — the budget is fixed.
        if (!grow_pool_for(R, R.inv->host_id(), n))
        {
            break;
        }
        NodeId const host = R.inv->host_id();
        R.inv->grow_node(host, R.host_pool->block_count());
        if (R.executor != nullptr)
        {
            R.executor->apply_growth(host, R.host_pool->block_count());
        }
        sync_event_provisions(R);
        blocks.grow_to(R.host_pool->block_count()); // tail free
    }
    if (!window.has_value())
    {
        throw Error("host acquire of tile \"" + label_of(impl) + "\" (" +
                    std::to_string(n) + " chunks): no contiguous run of "
                    "free-or-evictable host blocks remains (the host "
                    "pool is pinned or full)");
    }

    Plan plan;
    plan.plan_seq = R.next_plan_seq;
    plan.entry = R.seed;
    std::vector<SeqTask> seq;

    // The evictions first (with survival): the acquire's LINKs fold
    // over the freed world.
    for (ChunkId victim : window->victims)
    {
        EntryBinding const held = *blocks.held_[victim];
        bool const last_actual =
            held.actual && !world.actual_replica_elsewhere(
                               held.tile, held.index, host);
        if (!last_actual)
        {
            // Outdated, or another actual replica elsewhere: bare.
            plan.queue.push_back(
                make_unlink(held.tile, held.index, host, victim));
            world.unlink_binding(host, victim);
            blocks.free_[victim] = 1;
            blocks.held_[victim] = std::nullopt;
            continue;
        }
        // The rescue, the PLANNER'S OWN DISCIPLINE (the landing
        // order's choice, ported to the boundary act): an EXISTING
        // binding of the logical chunk on a device is the landing —
        // the COPY lands on the outdated one, no new binding (a fresh
        // link beside it would be the checker's double-binding) —
        // and only a node carrying NO binding of the chunk takes the
        // fresh-link form. Then the UNLINK.
        bool landed = false;
        for (std::size_t d = 0;
             !landed && d < R.inv->device_count(); ++d)
        {
            NodeId const dev = static_cast<NodeId>(d);
            auto ob = world.binding_of(held.tile, held.index, dev);
            if (ob.has_value())
            {
                // The premise holds: this binding is OUTDATED (the
                // victim is the last ACTUAL replica).
                plan.queue.push_back(make_copy(held.tile, held.index,
                    ChunkLoc{host, victim},
                    ChunkLoc{dev, ob->first}));
                world.link_binding(held.tile, held.index, dev,
                    ob->first, true);
                landed = true;
            }
        }
        if (!landed)
        {
            auto stage = free_device_landing(
                    R, world, plan, blocks, window->base, n);
            // GROW FIRST: a failed landing retries once
            // per grown chunk — each device pool may still grow to its
            // soft limit, and every fresh chunk is a landing.
            while (!stage.has_value() &&
                   grow_device_pools(R, 1))
            {
                stage = free_device_landing(
                    R, world, plan, blocks, window->base, n);
            }
            if (!stage.has_value())
            {
                throw acquire_census(R, world, plan,
                    "the eviction of tile " + std::to_string(held.tile) +
                        " chunk " + std::to_string(held.index) +
                        " (the last actual replica) found no device "
                        "landing to rescue it onto");
            }
            plan.queue.push_back(make_link(held.tile, held.index,
                stage->first, stage->second));
            plan.queue.push_back(make_copy(held.tile, held.index,
                ChunkLoc{host, victim},
                ChunkLoc{stage->first, stage->second}));
            world.link_binding(held.tile, held.index, stage->first,
                stage->second, true);
        }
        plan.queue.push_back(
            make_unlink(held.tile, held.index, host, victim));
        world.unlink_binding(host, victim);
        blocks.free_[victim] = 1;
        blocks.held_[victim] = std::nullopt;
    }

    // The tile's OWN host bindings (the one-binding-per-node property
    // — the double-binding defect the nntile core suite
    // caught): a binding AT its chunk's run position is REUSED (no
    // LINK, and for a Read acquire with actual bytes no COPY either);
    // any other binding of the tile leaves the host BEFORE the run's
    // LINKs — bare for a pure Write acquire (the user's rewrite
    // supersedes the bytes unconditionally), and with its bytes
    // repatriated onto a device otherwise (the same landing ladder as
    // the victims', the run's COPY sourcing the landing).
    struct OwnBinding
    {
        ChunkId block = 0;
        bool actual = false;
    };
    std::vector<std::optional<OwnBinding>> own(n);
    std::vector<bool> reuse(n, false);
    for (std::size_t i = 0; i < n; ++i)
    {
        auto hb = world.binding_of(
            impl.tile_id, static_cast<ChunkIndex>(i), host);
        if (hb.has_value())
        {
            own[i] = OwnBinding{hb->first, hb->second};
        }
        if (own[i].has_value() &&
            own[i]->block == static_cast<ChunkId>(window->base + i))
        {
            reuse[i] = true;
        }
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        if (!own[i].has_value() || reuse[i])
        {
            continue;
        }
        ChunkIndex const ix = static_cast<ChunkIndex>(i);
        if (access == Access::Write || !own[i]->actual)
        {
            // The bytes die here lawfully: a Write acquire's rewrite
            // supersedes them; an outdated binding holds none.
            plan.queue.push_back(
                make_unlink(impl.tile_id, ix, host, own[i]->block));
        }
        else
        {
            // Read/ReadWrite: the last bytes must reach the run — a
            // host-internal move is not a plan op, so
            // they ride to a device landing first (an existing binding
            // of the chunk refreshed, else a fresh landing), and the
            // run's COPY sources that replica.
            bool landed = false;
            for (std::size_t d = 0;
                 !landed && d < R.inv->device_count(); ++d)
            {
                NodeId const dev = static_cast<NodeId>(d);
                auto ob = world.binding_of(impl.tile_id, ix, dev);
                if (ob.has_value())
                {
                    plan.queue.push_back(make_copy(impl.tile_id, ix,
                        ChunkLoc{host, own[i]->block},
                        ChunkLoc{dev, ob->first}));
                    world.link_binding(
                        impl.tile_id, ix, dev, ob->first, true);
                    landed = true;
                }
            }
            if (!landed)
            {
                auto stage = free_device_landing(
                    R, world, plan, blocks, window->base, n);
                if (!stage.has_value())
                {
                    throw acquire_census(R, world, plan,
                        "the repatriation of tile " +
                            std::to_string(impl.tile_id) + " chunk " +
                            std::to_string(i) +
                            " (the last actual replica) found no device "
                            "landing to ride to the run");
                }
                plan.queue.push_back(make_link(impl.tile_id, ix,
                    stage->first, stage->second));
                plan.queue.push_back(make_copy(impl.tile_id, ix,
                    ChunkLoc{host, own[i]->block},
                    ChunkLoc{stage->first, stage->second}));
                world.link_binding(impl.tile_id, ix, stage->first,
                    stage->second, true);
            }
            plan.queue.push_back(
                make_unlink(impl.tile_id, ix, host, own[i]->block));
        }
        world.unlink_binding(host, own[i]->block);
        blocks.free_[own[i]->block] = 1;
        blocks.held_[own[i]->block] = std::nullopt;
    }

    if (access == Access::Write)
    {
        // The Write acquire hands the stable run out for a host
        // rewrite: LINK it (the bindings start OUTDATED — the user's
        // bytes are not the tile's version until the unpin publishes),
        // and the publish TILE_OP at the release marks them actual.
        // A LINK-only plan is legal (no task, no correspondence row).
        for (std::size_t i = 0; i < n; ++i)
        {
            if (reuse[i])
            {
                continue; // the run block already binds this chunk
            }
            plan.queue.push_back(make_link(impl.tile_id,
                static_cast<ChunkIndex>(i), host,
                static_cast<ChunkId>(window->base + i)));
        }
        run_plan(R, plan, seq);
        impl.pins += 1;
        R.acquired.emplace(&impl,
            AcquireRun{window->base, n,
                publish && !impl.dead.load(std::memory_order_relaxed)});
        return R.host_pool->addr(window->base);
    }

    // The Read/ReadWrite acquire: LINK the run, COPY each chunk from
    // its ACTUAL replica (the lowest device; a host-only actual
    // replica rides a device round-trip — a host-internal move is not
    // a plan op), and the host gather TILE_OP reading
    // the whole run last (the acquire IS a host-node
    // TILE_OP; the gather itself is the copies).
    for (std::size_t i = 0; i < n; ++i)
    {
        ChunkIndex const ix = static_cast<ChunkIndex>(i);
        ChunkId const dst =
            static_cast<ChunkId>(window->base + i);
        if (!reuse[i])
        {
            plan.queue.push_back(
                make_link(impl.tile_id, ix, host, dst));
        }
        else if (own[i]->actual)
        {
            // The run block already holds this chunk's ACTUAL bytes:
            // nothing to move (the acquire's whole point).
            continue;
        }

        // The source: the lowest DEVICE node holding an actual
        // replica in the post-eviction world (an own ACTUAL binding
        // was repatriated above, so this always finds one when the
        // data exists).
        auto dev_src = world.device_actual_replica(
            impl.tile_id, ix, R.inv->device_count());
        if (dev_src.has_value())
        {
            plan.queue.push_back(
                make_copy(impl.tile_id, ix, *dev_src, ChunkLoc{host, dst}));
            world.link_binding(impl.tile_id, ix, host, dst, true);
            continue;
        }

        // No actual replica anywhere (the own-binding pass above
        // repatriated every actual host binding onto a device, so a
        // miss means the predicted world never produced this data):
        // a caller defect, never serviced.
        throw Error("read of tile \"" + label_of(impl) +
                    "\" that has never been written");
    }

    // NO host TILE_OP (the old engine's
    // shape): a CONTIGUOUS run's gather is ZERO ops — the D2H copies
    // above ARE the whole acquire, their mints are CHANNEL mints
    // (device-side, stream-waited by any later consumer), and the
    // pinned base pointer is handed straight to the CPU (host RAM
    // access, never a lane op). The statement "a host acquire IS a
    // host-node TILE_OP" reads as applying when an actual scatter-
    // assembling gather exists — a contiguous run needs none, so no
    // host-lane USER fact ever forms here and no later device op owes
    // a host-family mint through this path.
    run_plan(R, plan, seq);
    impl.pins += 1;
    R.acquired.emplace(&impl,
        AcquireRun{window->base, n,
            publish && !impl.dead.load(std::memory_order_relaxed)});
    return R.host_pool->addr(window->base);
}

void engine_release(RuntimeState &R, HandleImpl &impl,
    std::uint64_t generation)
{
    if(generation != R.generation)
    {
        return; // an Acquired from a previous runtime generation
    }
    std::lock_guard<std::mutex> lock(R.mu);
    if (impl.pins > 0)
    {
        impl.pins -= 1;
    }
    if (impl.pins > 0)
    {
        // A stacked acquire still holds this run; keep its metadata
        // (notably a Write-pending publish) for the outermost release.
        return;
    }
    auto it = R.acquired.find(&impl);
    if (it == R.acquired.end())
    {
        return;
    }
    AcquireRun const run = it->second;
    bool const publish = run.publish &&
                         !R.error_set &&
                         !impl.dead.load(std::memory_order_relaxed);
    R.acquired.erase(it);
    if (!publish)
    {
        return;
    }

    // The publish plan: the host TILE_OP
    // with the tile's access Write over the run IS the publication
    // (the fold marks the run actual and outdates every other
    // replica; the body is the engine's no-op — protocol-irreducible:
    // only a Write operand flips a host binding's flag), and it is
    // followed by the RESTORE COPIES — H2D host->device for every
    // chunk whose device replica still stands. The restores are the
    // PREFETCH: the next batch reads the device replica and owes the
    // H2D COPY's mint — a CHANNEL mint, waited STREAM-SIDE — never
    // the publish op's host-family mint, so the steady loop carries
    // no host-lane debt at all (the old engine's mechanics: sync
    // between streams is the non-blocking stream wait, and the only
    // host block is the mini-plan's own terminal-event join).
    NodeId const host = R.inv->host_id();
    Plan plan;
    plan.plan_seq = R.next_plan_seq;
    plan.entry = R.seed;
    TaskId const pub = engine_host_act_symbol(R, 2);
    std::vector<Operand> ops;
    ops.reserve(run.n);
    for (std::size_t i = 0; i < run.n; ++i)
    {
        Operand o;
        o.tile = impl.tile_id;
        o.index = static_cast<ChunkIndex>(i);
        o.access = Access::Write;
        o.node = host;
        o.chunk = static_cast<ChunkId>(run.base + i);
        ops.push_back(o);
    }
    plan.queue.push_back(make_tile_op(pub, host, std::move(ops)));

    // NO restore copies (the mint record made them redundant): a later
    // reader's refetch COPY owes THIS publish's mint — a HOST-lane
    // mint whose TWIN is a device event the host worker records at
    // the op's FIFO position — and the consumer's wait is a plain
    // STREAM WAIT, never a deferral gate. Under
    // pool pressure the restored replica would be evicted and
    // refetched anyway: the restores only added traffic on top of the
    // planner's own churn, and the extra traffic slows an iteration.

    std::vector<SeqTask> seq;
    SeqTask st;
    st.task = pub;
    st.operands.push_back(TileAccess{impl.tile_id, Access::Write});
    seq.push_back(std::move(st));
    run_plan(R, plan, seq);
}

void engine_release_swept(RuntimeState &R, HandleImpl *impl)
{
    impl->dead_done = true;
    impl->swept_pending = false;
    R.dead_pending_unlink(impl);
    if(impl->engine_owned)
    {
        delete impl;
    }
}

bool engine_handle_referenced(RuntimeState &R, Handle const *handle)
{
    // Lock order matches engine_issue_batch: R.mu, then submit_mu.
    std::lock_guard<std::mutex> lock(R.mu);
    for (RuntimeState::IssuedBatch const &record : R.unretired)
    {
        for (Task *t = record.tasks; t != nullptr; t = t->next)
        {
            for (std::size_t i = 0; i < t->nbufs; ++i)
            {
                if (t->bufs()[i].handle == handle)
                {
                    return true;
                }
            }
        }
    }
    std::lock_guard<std::mutex> submit_lock(R.submit_mu);
    for (Task *t = R.queue.pending_head_.load(std::memory_order_relaxed);
         t != nullptr; t = t->next)
    {
        for (std::size_t i = 0; i < t->nbufs; ++i)
        {
            if (t->bufs()[i].handle == handle)
            {
                return true;
            }
        }
    }
    return false;
}

void engine_note_retire(RuntimeState &R)
{
    // The previous engine's dedicated non-closing, non-draining
    // retirement note, carried VERBATIM in shape (its 8.3: "never a
    // plan, never a join, never the depth check"): the note is a pure
    // MARKER — the caller already appended the handle to dead-pending
    // — and the retirement itself rides the NEXT close (a batch
    // close's dead tiles and dying frees, the planner's dying-free
    // order, or an idle close's one retire-only sweep over the whole
    // accumulated set). A destructor is noexcept, and destructors
    // arrive in BURSTS (a recorder's gc drops a whole session's
    // temporaries back-to-back): a per-death close would issue one
    // tiny retire plan per handle — the multi-second tail waits seen
    // in the pipelined loop — so the note closes nothing, ever.
    (void) R;
}

// ---- the accounting and the simulate exposure ----

void engine_note_batch(RuntimeState &R, Plan const &plan,
    PlanExecutor::PlanIssue const &issue)
{
    std::size_t position = 0;
    for (PlanOp const &op : issue.issue_log)
    {
        switch (op.kind)
        {
        case PlanOpKind::Link:
            R.link_ops += 1;
            break;
        case PlanOpKind::Unlink:
            R.unlink_ops += 1;
            break;
        case PlanOpKind::Copy:
            R.copy_ops += 1;
            {
                CopyRecord rec;
                rec.op_ref = make_op_ref(plan.plan_seq, position);
                auto it = R.live_tiles.find(op.u.copy.tile);
                rec.handle = it != R.live_tiles.end()
                    ? label_of(*it->second)
                    : ("tile " + std::to_string(op.u.copy.tile));
                rec.src_node = op.u.copy.src.node;
                rec.dst_node = op.u.copy.dst.node;
                // The record's bytes: the COPY's occupied span — the
                // SAME derivation the dispatch issues (tile_chunk_span
                // over the registered payload
                // bytes), never a class-constant stamp: a tail chunk's
                // record carries its payload remainder, a full chunk's
                // the whole class.
                rec.nbytes = tile_chunk_span(
                    R.tiles.nbytes(op.u.copy.tile), op.u.copy.index);
                R.copy_log_.push_back(std::move(rec));
            }
            break;
        case PlanOpKind::TileOp:
            R.tile_ops += 1;
            break;
        }
        ++position;
    }
}

void engine_simulate_plan(RuntimeState &R, Plan const &plan,
    PlanDerivation const &deriv)
{
    // Completeness first (a miss belongs to an explicit simulate's
    // census; the engine's exposure simply does not advance on a
    // miss): every TILE_OP's (codelet, node) duration and every COPY's
    // channel rate must be modeled.
    for (PlanOp const &op : plan.queue)
    {
        if (op.kind == PlanOpKind::TileOp)
        {
            auto tc = R.task_codelet.find(op.u.tile_op.task);
            if (tc == R.task_codelet.end())
            {
                return;
            }
            std::uint64_t const key =
                (static_cast<std::uint64_t>(tc->second) << 8) |
                op.u.tile_op.node;
            if (R.duration_by_codelet.find(key) ==
                R.duration_by_codelet.end())
            {
                return;
            }
        }
        else if (op.kind == PlanOpKind::Copy)
        {
            LaneId const lane = R.inv->lane_of_chan(
                op.u.copy.src.node, op.u.copy.dst.node);
            if (lane == kNoLane ||
                static_cast<std::size_t>(lane) >=
                    R.channel_rates.size() ||
                R.channel_rates[lane] == 0)
            {
                return;
            }
        }
    }

    // The owed origins must be timed too: a plan whose owed sets name
    // an op of a SKIPPED plan skips as well — its start rule would
    // demand a simulated end that was never registered (the chunk's
    // USER fact never resets, so freed-and-reused chunks
    // keep owing their last producers across plans).
    for (std::vector<OpRef> const &owed : deriv.owed_set)
    {
        for (OpRef ref : owed)
        {
            std::uint64_t const origin = ref >> 32;
            if (origin != plan.plan_seq &&
                R.simulated_plans.find(origin) == R.simulated_plans.end())
            {
                return;
            }
        }
    }

    // The instance-keyed view (the model's consumers name instances;
    // the loading surface names codelets).
    R.sim_model.task_node_ns.clear();
    R.sim_model.channel_bytes_per_ns = R.channel_rates;
    for (PlanOp const &op : plan.queue)
    {
        if (op.kind != PlanOpKind::TileOp)
        {
            continue;
        }
        std::uint32_t const codelet = R.task_codelet[op.u.tile_op.task];
        std::uint64_t const ckey =
            (static_cast<std::uint64_t>(codelet) << 8) |
            op.u.tile_op.node;
        R.sim_model.task_node_ns[(
            static_cast<std::uint64_t>(op.u.tile_op.task) << 8) |
            op.u.tile_op.node] = R.duration_by_codelet[ckey];
    }
    if (!R.sim_pass)
    {
        R.sim_pass = std::make_unique<TimingPass>(*R.inv, R.sim_model);
    }
    R.sim_pass->run_plan(plan, deriv.owed_set);

    // The retire seam: the live refs are every op a current fact of
    // the dependency state names (a USER fact or a PRODUCER fact; the
    // USER fact never resets).
    std::vector<OpRef> live;
    DependencyState const &dep = R.executor->dep_state();
    for (OpRef r : dep.producer_)
    {
        if (r != kNoRef)
        {
            live.push_back(r);
        }
    }
    for (OpRef r : dep.user_)
    {
        if (r != kNoRef)
        {
            live.push_back(r);
        }
    }
    R.sim_pass->retire(live);
    R.simulate_makespan_ns = R.sim_pass->run_makespan();
    R.simulated_plans.insert(plan.plan_seq);
}

} // namespace nnhaul
