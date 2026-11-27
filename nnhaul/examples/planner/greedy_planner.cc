/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file examples/planner/greedy_planner.cc
 * A minimal external planner written ONLY against <nnhaul/planner.hh>.
 *
 * Rules it obeys (the same the engine's checker enforces):
 *   - one binding per (node, logical chunk); never allocate a second one;
 *   - a COPY needs BOTH endpoints bound, src actual, dst not actual;
 *   - a Write needs a bound chunk, makes it actual, and OUTDATES every
 *     other replica of that logical chunk;
 *   - a Read/ReadWrite operand must be actual: copy into the existing
 *     outdated local binding when one exists, else allocate + LINK +
 *     COPY;
 *   - chunk ids are valid below pool_chunks(); grow() raises that and is
 *     recorded before use;
 *   - the exit residency reports the post-batch table.
 *
 * The planner object is process-lifetime: every call starts from an
 * empty local view.
 *
 * SPDX-License-Identifier: MIT
 *
 * @version 1.0.0
 * */

#include <nnhaul/planner.hh>

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using namespace nnhaul;
using namespace nnhaul::planner;
using Lc = std::pair<TileId, ChunkIndex>;

struct Res
{
    NodeId node = 0;
    ChunkId chunk = 0;
    bool actual = false;
};

class Greedy final: public Planner
{
  public:
    char const *name() const override { return "example.greedy"; }

    void plan(PlanRequest const &request, Topology const &topology,
        Sink &sink) override
    {
        std::lock_guard<std::mutex> const lock(mu_);
        topology_ = &topology;
        sink_ = &sink;
        bind_.clear();
        cardinality_.assign(topology.node_count(), 0);
        free_.assign(topology.node_count(), {});
        for(std::size_t n = 0; n < topology.node_count(); ++n)
        {
            cardinality_[n] = topology.pool_chunks(static_cast<NodeId>(n));
            for(std::size_t c = 0; c < cardinality_[n]; ++c)
            {
                free_[n].insert(static_cast<ChunkId>(c));
            }
        }
        for(EntryBinding const &b : request.entry)
        {
            bind_[{b.tile, b.index}].push_back({b.node, b.chunk, b.actual});
            free_.at(b.node).erase(b.chunk);
        }

        for(PlannedTask const &task : request.tasks)
        {
            // The launch contract is buffer-major, chunks ascending: the
            // engine passes one pointer run per submitted buffer, in the
            // order the buffers appear in the task. Emit exactly that.
            std::vector<Operand> operands;
            for(TileUse const &use : task.operands)
            {
                ChunkIndex const n = topology.tile_chunks(use.tile);
                for(ChunkIndex i = 0; i < n; ++i)
                {
                    Res const r = resolve(use.tile, i, task.node,
                        use.access);
                    operands.push_back({use.tile, i, use.access,
                        r.node, r.chunk});
                }
            }
            sink.tile_op(task.task, task.node, operands);
            for(Operand const &op : operands)
            {
                if(op.access != Access::Read)
                {
                    Lc const lc{op.tile, op.index};
                    set_actual(lc, op.node, op.chunk);
                    outdate_others(lc, op.node, op.chunk);
                }
            }
        }

        retire_dead(request.dead_tiles);
        emit_exit(request, sink);
    }

  private:
    Res *find(Lc lc, NodeId node)
    {
        auto it = bind_.find(lc);
        if(it == bind_.end())
        {
            return nullptr;
        }
        for(Res &r : it->second)
        {
            if(r.node == node)
            {
                return &r;
            }
        }
        return nullptr;
    }

    Res const *find_actual_any(Lc lc) const
    {
        auto it = bind_.find(lc);
        if(it == bind_.end())
        {
            return nullptr;
        }
        for(Res const &r : it->second)
        {
            if(r.actual)
            {
                return &r;
            }
        }
        return nullptr;
    }

    ChunkId alloc(NodeId node)
    {
        auto &free = free_.at(node);
        if(!free.empty())
        {
            ChunkId const c = *free.begin();
            free.erase(free.begin());
            return c;
        }
        std::size_t const limit = topology_->pool_capacity(node);
        if(limit != 0 && cardinality_[node] >= limit)
        {
            throw std::runtime_error(
                "example.greedy: the node's chunk pool is at its soft limit");
        }
        cardinality_[node] += 1;
        sink_->grow(node, cardinality_[node]);
        return static_cast<ChunkId>(cardinality_[node] - 1);
    }

    void link(Lc lc, NodeId node, ChunkId chunk, bool actual)
    {
        sink_->link(lc.first, lc.second, node, chunk);
        bind_[lc].push_back({node, chunk, actual});
    }

    void set_actual(Lc lc, NodeId node, ChunkId chunk)
    {
        for(Res &r : bind_[lc])
        {
            if(r.node == node && r.chunk == chunk)
            {
                r.actual = true;
                return;
            }
        }
    }

    void outdate_others(Lc lc, NodeId node, ChunkId chunk)
    {
        for(Res &r : bind_[lc])
        {
            if(r.node != node || r.chunk != chunk)
            {
                r.actual = false;
            }
        }
    }

    //! Give the task a binding for the logical chunk that is legal for
    //! its access mode. Copies are emitted, never simulated.
    Res resolve(TileId tile, ChunkIndex index, NodeId node, Access access)
    {
        Lc const lc{tile, index};
        Res *local = find(lc, node);
        if(local != nullptr && (access == Access::Write || local->actual))
        {
            return *local;
        }
        if(access == Access::Write)
        {
            if(local != nullptr)
            {
                return *local;
            }
            ChunkId const c = alloc(node);
            link(lc, node, c, false);
            return {node, c, false};
        }
        Res const *src = find_actual_any(lc);
        if(src == nullptr)
        {
            throw std::runtime_error(
                "example.greedy: a read has no actual replica anywhere");
        }
        // Copy the coordinates out: alloc()/link() may reallocate the
        // binding vector, so no pointer into it survives.
        NodeId const src_node = src->node;
        ChunkId const src_chunk = src->chunk;
        if(src_node == node)
        {
            return {node, src_chunk, true};
        }
        ChunkId dst_chunk;
        if(local != nullptr)
        {
            // Reuse the outdated local binding: a second binding of the
            // same logical chunk on one node is a checker fault.
            dst_chunk = local->chunk;
        }
        else
        {
            dst_chunk = alloc(node);
            link(lc, node, dst_chunk, false);
        }
        sink_->copy(tile, index, {src_node, src_chunk}, {node, dst_chunk});
        set_actual(lc, node, dst_chunk);
        return {node, dst_chunk, true};
    }

    void retire_dead(std::vector<TileId> const &dead)
    {
        for(TileId tile : dead)
        {
            for(auto it = bind_.begin(); it != bind_.end();)
            {
                if(it->first.first != tile)
                {
                    ++it;
                    continue;
                }
                for(Res const &r : it->second)
                {
                    sink_->unlink(tile, it->first.second, r.node, r.chunk);
                    free_.at(r.node).insert(r.chunk);
                }
                it = bind_.erase(it);
            }
        }
    }

    void emit_exit(PlanRequest const &request, Sink &sink) const
    {
        std::vector<EntryBinding> exit;
        for(auto const &kv : bind_)
        {
            for(Res const &r : kv.second)
            {
                exit.push_back({r.node, r.chunk, kv.first.first,
                    kv.first.second, r.actual});
            }
        }
        std::sort(exit.begin(), exit.end(),
            [](EntryBinding const &a, EntryBinding const &b) {
                return a.node != b.node ? a.node < b.node : a.chunk < b.chunk;
            });
        sink.set_exit(exit);
        PlanStats stats;
        stats.entry_bindings = request.entry.size();
        stats.exit_bindings = exit.size();
        sink.set_stats(stats);
    }

    // The planner object is process-lifetime and may be shared by two
    // Runtime instances; plan() serializes its own mutable view.
    std::mutex mu_;
    Topology const *topology_ = nullptr;
    Sink *sink_ = nullptr;
    std::map<Lc, std::vector<Res>> bind_;
    std::vector<std::size_t> cardinality_;
    std::vector<std::set<ChunkId>> free_;
};

Greedy g_planner;

} // namespace

extern "C" std::uint32_t nnhaul_planner_api_version()
{
    return nnhaul::planner::kApiVersion;
}

extern "C" nnhaul::planner::Planner *nnhaul_create_planner()
{
    return &g_planner;
}
