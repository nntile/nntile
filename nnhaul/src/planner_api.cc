/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/planner_api.cc
 * The public planner interface's engine side: a read-only topology view,
 * a sink that builds the engine's plan, the in-tree LRU path, and the
 * NNHAUL_PLANNER dynamic loader.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "planner_api.hh"

#include "inventory.hh"
#include "plan.hh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

extern "C" void const *nnhaul_engine_identity();

namespace nnhaul::planner
{

namespace
{

class TopologyImpl final: public Topology
{
  public:
    TopologyImpl(NodeInventory const &inv, TileTable const &tiles,
        std::vector<std::uint64_t> const &rates,
        std::map<NodeId, std::size_t> const &grown):
        inv_(inv), tiles_(tiles), rates_(rates), grown_(grown)
    {
    }

    std::size_t node_count() const override { return inv_.node_count(); }
    bool is_host(NodeId node) const override { return inv_.is_host(node); }
    std::size_t pool_capacity(NodeId node) const override
    {
        return inv_.pool_limit(node);
    }
    std::size_t pool_chunks(NodeId node) const override
    {
        std::size_t n = inv_.pool_cardinality(node);
        auto it = grown_.find(node);
        if(it != grown_.end() && it->second > n)
        {
            n = it->second;
        }
        return n;
    }
    bool chunk_valid(NodeId node, ChunkId chunk) const override
    {
        return static_cast<std::size_t>(chunk) < pool_chunks(node);
    }
    bool has_channel(NodeId src, NodeId dst) const override
    {
        return inv_.has_channel(src, dst);
    }
    double channel_rate(NodeId src, NodeId dst) const override
    {
        if(!inv_.has_channel(src, dst))
        {
            return 0.0;
        }
        std::size_t const lane = inv_.lane_of_chan(src, dst);
        if(lane >= rates_.size())
        {
            return 1.0;
        }
        // A registered channel without a measured rate is neutral.
        return rates_[lane] == 0
            ? 1.0 : static_cast<double>(rates_[lane]);
    }
    bool carries(TileId tile) const override { return tiles_.carries(tile); }
    ChunkIndex tile_chunks(TileId tile) const override
    {
        return tiles_.n(tile);
    }
    std::size_t tile_bytes(TileId tile) const override
    {
        return tiles_.nbytes(tile);
    }

    NodeInventory const &inventory() const { return inv_; }
    TileTable const &tiles() const { return tiles_; }

  private:
    NodeInventory const &inv_;
    TileTable const &tiles_;
    std::vector<std::uint64_t> const &rates_;
    std::map<NodeId, std::size_t> const &grown_;
};

class SinkImpl final: public Sink
{
  public:
    SinkImpl(PlannedBatch &out, NodeInventory const &inv,
        std::map<NodeId, std::size_t> &grown):
        out_(out), inv_(inv), grown_(grown)
    {
    }

    bool exit_set() const { return exit_set_; }

    void link(TileId tile, ChunkIndex index, NodeId node,
        ChunkId chunk) override
    {
        PlanOp op;
        op.kind = PlanOpKind::Link;
        op.u.link = {tile, index, node, chunk};
        out_.plan.queue.push_back(std::move(op));
    }
    void unlink(TileId tile, ChunkIndex index, NodeId node,
        ChunkId chunk) override
    {
        PlanOp op;
        op.kind = PlanOpKind::Unlink;
        op.u.unlink = {tile, index, node, chunk};
        out_.plan.queue.push_back(std::move(op));
    }
    void copy(TileId tile, ChunkIndex index, ChunkLoc src,
        ChunkLoc dst) override
    {
        PlanOp op;
        op.kind = PlanOpKind::Copy;
        op.u.copy = {tile, index, src, dst};
        out_.plan.queue.push_back(std::move(op));
    }
    void tile_op(TaskId task, NodeId node,
        std::vector<Operand> const &operands) override
    {
        PlanOp op;
        op.kind = PlanOpKind::TileOp;
        op.u.tile_op = {task, node};
        op.operands = operands;
        out_.plan.queue.push_back(std::move(op));
    }
    void grow(NodeId node, std::size_t cardinality) override
    {
        if(static_cast<std::size_t>(node) >= inv_.node_count())
        {
            throw Error("planner: grow() names a node outside the inventory");
        }
        std::size_t current = inv_.pool_cardinality(node);
        auto it = grown_.find(node);
        if(it != grown_.end() && it->second > current)
        {
            current = it->second;
        }
        if(cardinality <= current)
        {
            throw Error("planner: grow() must raise the pool cardinality");
        }
        std::size_t const limit = inv_.pool_limit(node);
        if(limit != 0 && cardinality > limit)
        {
            throw Error("planner: grow() exceeds the pool's soft limit");
        }
        grown_[node] = cardinality;
        if(out_.plan.growth.size() <= node)
        {
            out_.plan.growth.resize(node + 1, 0);
        }
        out_.plan.growth[node] = cardinality;
    }
    void set_exit(std::vector<EntryBinding> const &exit) override
    {
        out_.exit = exit;
        exit_set_ = true;
    }
    void set_stats(PlanStats const &stats) override
    {
        auto &d = out_.stats;
        d.evictions = stats.evictions;
        d.rescues = stats.rescues;
        d.refetches = stats.refetches;
        d.prefetches = stats.prefetches;
        d.dying_frees = stats.dying_frees;
        d.entry_bindings = stats.entry_bindings;
        d.exit_bindings = stats.exit_bindings;
        d.run_misses = stats.run_misses;
        d.run_miss_want = stats.run_miss_want;
        d.run_miss_longest = stats.run_miss_longest;
        d.pin_spends = stats.pin_spends;
    }

  private:
    PlannedBatch &out_;
    NodeInventory const &inv_;
    std::map<NodeId, std::size_t> &grown_;
    bool exit_set_ = false;
};

} // namespace

namespace
{

//! One dlopen and one create per path per process. Every Runtime that
//! names the same NNHAUL_PLANNER shares the planner object; the handle
//! stays open for the process lifetime because the object lives in the
//! plugin. (A plugin may create a fresh object per call; that object is
//! owned by the plugin for the process lifetime.)
std::mutex &load_mutex()
{
    static std::mutex mu;
    return mu;
}

std::map<std::string, Planner *> &loaded_planners()
{
    static std::map<std::string, Planner *> loaded;
    return loaded;
}

Planner *load_once(char const *path)
{
    void *handle = ::dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if(handle == nullptr)
    {
        throw Error(std::string("NNHAUL_PLANNER: cannot load '")
            + path + "': " + ::dlerror());
    }
    // A plugin built against the static archive carries its own engine;
    // issuing its plan through this runtime would violate the one-engine
    // contract. Compare the process token before anything else runs.
    bool engine_ok = false;
    if(void *sym = ::dlsym(handle, "nnhaul_engine_identity"))
    {
        auto identity_fn = reinterpret_cast<void const *(*)()>(sym);
        engine_ok = identity_fn() == nnhaul_engine_identity();
    }
    // A static host has a null token on both sides; compare the actual
    // chunk_class implementation instead. A plugin that pulled its own
    // static engine has a different function address.
    if(void *cc = ::dlsym(handle, "_ZN6nnhaul11chunk_classEv"))
    {
        if(reinterpret_cast<void *>(cc)
            != reinterpret_cast<void *>(&::nnhaul::chunk_class))
        {
            engine_ok = false;
            throw Error(std::string("NNHAUL_PLANNER: '") + path
                + "' contains its own static nnhaul engine; link the "
                  "same engine build as the host");
        }
    }
    // A pre-identity static build still exports the engine's runtime
    // singleton. If it is there without our token, the plugin carries
    // its own engine; a header-only planner exports neither and is fine.
    if(!engine_ok && ::dlsym(handle, "_ZN6nnhaul2rtE") != nullptr)
    {
        throw Error(std::string("NNHAUL_PLANNER: '") + path
            + "' contains its own static nnhaul engine; link the "
              "shared libnnhaul instead");
    }
    auto version_fn = reinterpret_cast<nnhaul_planner_api_version_fn>(
        ::dlsym(handle, "nnhaul_planner_api_version"));
    auto create_fn = reinterpret_cast<nnhaul_planner_create_fn>(
        ::dlsym(handle, "nnhaul_create_planner"));
    if(version_fn == nullptr || create_fn == nullptr)
    {
        throw Error(std::string("NNHAUL_PLANNER: '") + path
            + "' does not export nnhaul_planner_api_version and "
              "nnhaul_create_planner");
    }
    std::uint32_t version = 0;
    try
    {
        version = version_fn();
    }
    catch(...)
    {
        throw Error(std::string("NNHAUL_PLANNER: '") + path
            + "' threw from nnhaul_planner_api_version");
    }
    if(version != kApiVersion)
    {
        throw Error(std::string("NNHAUL_PLANNER: '") + path
            + "' implements planner API version "
            + std::to_string(version) + ", this library expects "
            + std::to_string(kApiVersion));
    }
    Planner *p = nullptr;
    try
    {
        p = create_fn();
    }
    catch(std::exception const &e)
    {
        throw Error(std::string("NNHAUL_PLANNER: '") + path
            + "' threw from nnhaul_create_planner: " + e.what());
    }
    catch(...)
    {
        throw Error(std::string("NNHAUL_PLANNER: '") + path
            + "' threw a non-standard exception from nnhaul_create_planner");
    }
    if(p == nullptr)
    {
        throw Error(std::string("NNHAUL_PLANNER: '") + path
            + "' returned no planner");
    }
    return p;
}

} // namespace

Planner *load(char const *path)
{
    if(path == nullptr || *path == '\0')
    {
        throw Error("planner load: empty path");
    }
    std::string const key(path);
    std::lock_guard<std::mutex> const lock(load_mutex());
    auto &loaded = loaded_planners();
    auto const it = loaded.find(key);
    if(it != loaded.end())
    {
        return it->second;
    }
    Planner *const p = load_once(path);
    loaded.emplace(key, p);
    return p;
}

Planner *load_from_env()
{
    char const *path = std::getenv("NNHAUL_PLANNER");
    if(path == nullptr || *path == '\0')
    {
        return nullptr;
    }
    return load(path);
}

namespace
{

//! The in-tree LRU planner as an ordinary consumer of the public
//! interface: it plans against the concrete world view and hands its
//! queue back through the same Sink a plugin writes to.
class InTreeLru final: public Planner
{
  public:
    char const *name() const override { return "nnhaul.lru"; }

    void plan(PlanRequest const &request, Topology const &topology,
        Sink &sink) override
    {
        auto const &topo = dynamic_cast<TopologyImpl const &>(topology);
        LruPlanner lru;
        PlannedBatch pb = lru.plan(request, topo.inventory(), topo.tiles());
        for(PlanOp const &op : pb.plan.queue)
        {
            switch(op.kind)
            {
                case PlanOpKind::Link:
                    sink.link(op.u.link.tile, op.u.link.index,
                        op.u.link.node, op.u.link.chunk);
                    break;
                case PlanOpKind::Unlink:
                    sink.unlink(op.u.unlink.tile, op.u.unlink.index,
                        op.u.unlink.node, op.u.unlink.chunk);
                    break;
                case PlanOpKind::Copy:
                    sink.copy(op.u.copy.tile, op.u.copy.index,
                        op.u.copy.src, op.u.copy.dst);
                    break;
                case PlanOpKind::TileOp:
                    sink.tile_op(op.u.tile_op.task, op.u.tile_op.node,
                        op.operands);
                    break;
            }
        }
        if(pb.plan.growth.size() > 0)
        {
            for(std::size_t n = 0; n < pb.plan.growth.size(); ++n)
            {
                if(pb.plan.growth[n] != 0)
                {
                    sink.grow(static_cast<NodeId>(n), pb.plan.growth[n]);
                }
            }
        }
        sink.set_exit(pb.exit);
        PlanStats stats;
        stats.evictions = pb.stats.evictions;
        stats.rescues = pb.stats.rescues;
        stats.refetches = pb.stats.refetches;
        stats.prefetches = pb.stats.prefetches;
        stats.dying_frees = pb.stats.dying_frees;
        stats.entry_bindings = pb.stats.entry_bindings;
        stats.exit_bindings = pb.stats.exit_bindings;
        stats.run_misses = pb.stats.run_misses;
        stats.run_miss_want = pb.stats.run_miss_want;
        stats.run_miss_longest = pb.stats.run_miss_longest;
        stats.pin_spends = pb.stats.pin_spends;
        sink.set_stats(stats);
    }
};

} // namespace

PlannedBatch run(Planner *plugin, PlanRequest const &request,
    NodeInventory const &inventory, TileTable const &tiles,
    std::vector<std::uint64_t> const &channel_rates)
{
    InTreeLru lru;
    Planner *chosen = plugin != nullptr ? plugin : &lru;
    PlannedBatch out;
    out.plan.plan_seq = request.plan_seq;
    out.plan.entry = request.entry;
    std::map<NodeId, std::size_t> grown;
    TopologyImpl topology(inventory, tiles, channel_rates, grown);
    SinkImpl sink(out, inventory, grown);
    chosen->plan(request, topology, sink);
    if(!sink.exit_set())
    {
        // The exit residency seeds the NEXT batch; a planner that omits
        // it would corrupt the following plan. Fail loudly, naming the
        // planner, rather than chain an empty entry state.
        throw Error(std::string("planner '") + chosen->name()
            + "' did not set the exit residency (Sink::set_exit)");
    }
    // The exit seeds the next batch and drives dead-tile retirement.
    std::set<std::pair<NodeId, ChunkId>> seen_chunk;
    std::set<std::tuple<NodeId, TileId, ChunkIndex>> seen_logical;
    for(EntryBinding const &b : out.exit)
    {
        bool const node_ok = static_cast<std::size_t>(b.node)
            < inventory.node_count();
        std::size_t chunks = 0;
        if(node_ok)
        {
            chunks = inventory.pool_cardinality(b.node);
            auto it = grown.find(b.node);
            if(it != grown.end() && it->second > chunks)
            {
                chunks = it->second;
            }
        }
        bool const ok = node_ok && tiles.carries(b.tile)
            && b.index < tiles.n(b.tile)
            && static_cast<std::size_t>(b.chunk) < chunks
            && seen_chunk.insert({b.node, b.chunk}).second
            && seen_logical.insert({b.node, b.tile, b.index}).second;
        if(!ok)
        {
            throw Error(std::string("planner '") + chosen->name()
                + "' emitted an invalid exit residency (unknown node, "
                  "tile, chunk, index, or a duplicate)");
        }
    }
    return out;
}

} // namespace nnhaul::planner
