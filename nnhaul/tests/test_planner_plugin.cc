/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file tests/test_planner_plugin.cc
 * The public planner interface, end to end at the PLANNING level: load the
 * example planner through nnhaul::planner::load(), plan a batch, then run
 * the engine's own checker on the emitted plan. No GPU is needed, and the
 * checker is the same one the engine runs before execution — so this test
 * fails if the example emits a plan the engine would reject.
 *
 * SPDX-License-Identifier: MIT
 *
 * @version 2.0.0
 * */

#include <nnhaul/planner.hh>

#include "inventory.hh"
#include "plan.hh"
#include "plan_checker.hh"

#include <cstdio>
#include <cstring>
#include <vector>

namespace
{

using namespace nnhaul;

int failures = 0;

#define CHECK(cond) do { if(!(cond)) { \
    std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, \
        #cond); ++failures; } } while(0)

class FakeTopology final: public planner::Topology
{
  public:
    FakeTopology(TileTable const &tiles): tiles_(tiles) {}
    std::size_t node_count() const override { return 2; }
    bool is_host(NodeId node) const override { return node == 1; }
    std::size_t pool_capacity(NodeId) const override { return 64; }
    std::size_t pool_chunks(NodeId) const override { return 8; }
    bool chunk_valid(NodeId, ChunkId chunk) const override
    {
        return chunk < 8;
    }
    bool has_channel(NodeId a, NodeId b) const override { return a != b; }
    double channel_rate(NodeId a, NodeId b) const override
    {
        return a != b ? 1.0 : 0.0;
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

  private:
    TileTable const &tiles_;
};

//! A sink that builds a real Plan, so the checker can judge it.
class PlanSink final: public planner::Sink
{
  public:
    void link(TileId t, ChunkIndex i, NodeId n, ChunkId c) override
    {
        PlanOp op;
        op.kind = PlanOpKind::Link;
        op.u.link = {t, i, n, c};
        plan.queue.push_back(op);
    }
    void unlink(TileId t, ChunkIndex i, NodeId n, ChunkId c) override
    {
        PlanOp op;
        op.kind = PlanOpKind::Unlink;
        op.u.unlink = {t, i, n, c};
        plan.queue.push_back(op);
    }
    void copy(TileId t, ChunkIndex i, ChunkLoc s, ChunkLoc d) override
    {
        PlanOp op;
        op.kind = PlanOpKind::Copy;
        op.u.copy = {t, i, s, d};
        plan.queue.push_back(op);
    }
    void tile_op(TaskId task, NodeId node,
        std::vector<Operand> const &ops) override
    {
        PlanOp op;
        op.kind = PlanOpKind::TileOp;
        op.u.tile_op = {task, node};
        op.operands = ops;
        plan.queue.push_back(op);
    }
    void grow(NodeId node, std::size_t cardinality) override
    {
        if(plan.growth.size() <= node)
        {
            plan.growth.resize(node + 1, 0);
        }
        plan.growth[node] = cardinality;
    }
    void set_exit(std::vector<EntryBinding> const &e) override { exit = e; }
    void set_stats(PlanStats const &) override {}

    Plan plan;
    std::vector<EntryBinding> exit;
};

} // namespace

int main()
{
    planner::Planner *p = nullptr;
    try
    {
        p = planner::load(NNHAUL_EXAMPLE_PLANNER);
    }
    catch(std::exception const &e)
    {
        std::fprintf(stderr, "load: %s\n", e.what());
        return 2;
    }
    CHECK(p != nullptr);
    if(p == nullptr)
    {
        return 1;
    }
    CHECK(std::strcmp(p->name(), "example.greedy") == 0);
    CHECK(planner::kApiVersion == 1);

    // Three tiles of two chunks each at the live chunk class.
    std::size_t const class_bytes = chunk_class_bytes();
    TileTable tiles;
    for(TileId t : {TileId{1}, TileId{2}, TileId{3}})
    {
        tiles.add(t, class_bytes + 1);
    }

    PlanRequest req;
    req.plan_seq = 1;
    for(TileId t : {TileId{1}, TileId{2}})
    {
        for(ChunkIndex i = 0; i < tiles.n(t); ++i)
        {
            req.entry.push_back({0, ChunkId(t * 2 + i), t, i, true});
        }
    }
    PlannedTask task;
    task.task = 7;
    task.node = 0;
    task.operands = {{1, Access::Read}, {2, Access::Read},
        {3, Access::Write}};
    req.tasks.push_back(task);

    NodeInventory inv(1, {8, 8}, {0});
    FakeTopology topo(tiles);
    PlanSink sink;
    p->plan(req, topo, sink);
    sink.plan.plan_seq = req.plan_seq;
    sink.plan.entry = req.entry;

    for(std::size_t n = 0; n < sink.plan.growth.size(); ++n)
    {
        if(sink.plan.growth[n] != 0)
        {
            inv.grow_node(static_cast<NodeId>(n), sink.plan.growth[n]);
        }
    }

    ExecutabilityTable runnable;
    runnable.add(7, {0});
    std::vector<SeqTask> seq{{7, {{1, Access::Read}, {2, Access::Read},
        {3, Access::Write}}}};
    CheckInput in;
    in.plan = &sink.plan;
    in.inventory = &inv;
    in.tiles = &tiles;
    in.runnable = &runnable;
    in.task_sequence = &seq;
    CheckResult const verdict = check(in);
    if(!verdict.pass)
    {
        std::fprintf(stderr, "checker rejected the example plan: %s at %zu\n",
            verdict.fault.c_str(), verdict.position);
        return 1;
    }
    CHECK(!sink.exit.empty());

    auto verify = [&](PlanRequest const &r, std::vector<SeqTask> const &seq,
        ExecutabilityTable &run, char const *what) {
        PlanSink out;
        try
        {
            p->plan(r, topo, out);
        }
        catch(std::exception const &e)
        {
            std::fprintf(stderr, "%s threw: %s\n", what, e.what());
            failures++;
            return;
        }
        out.plan.plan_seq = r.plan_seq;
        out.plan.entry = r.entry;
        for(std::size_t n = 0; n < out.plan.growth.size(); ++n)
        {
            if(out.plan.growth[n] != 0)
            {
                inv.grow_node(static_cast<NodeId>(n), out.plan.growth[n]);
            }
        }
        CheckInput ci = in;
        ci.plan = &out.plan;
        ci.runnable = &run;
        ci.task_sequence = &seq;
        CheckResult const v = check(ci);
        if(!v.pass)
        {
            std::fprintf(stderr, "%s rejected: %s at %zu\n", what,
                v.fault.c_str(), v.position);
            failures++;
        }
    };

    // Cross-node COPY: actual on the host, task on device 0.
    {
        PlanRequest r;
        r.plan_seq = 3;
        for(ChunkIndex i = 0; i < tiles.n(1); ++i)
        {
            r.entry.push_back({1, ChunkId(i), 1, i, true});
        }
        PlannedTask t;
        t.task = 9;
        t.node = 0;
        t.operands = {{1, Access::Read}};
        r.tasks.push_back(t);
        ExecutabilityTable run;
        run.add(9, {0});
        std::vector<SeqTask> seq{{9, {{1, Access::Read}}}};
        verify(r, seq, run, "cross-node");
    }

    // Write on node 0 then read on node 1: the write must outdate the
    // other replica, so the read copies the new version.
    {
        PlanRequest r;
        r.plan_seq = 4;
        for(ChunkIndex i = 0; i < tiles.n(3); ++i)
        {
            r.entry.push_back({0, ChunkId(6 + i), 3, i, true});
        }
        PlannedTask w;
        w.task = 10;
        w.node = 0;
        w.operands = {{3, Access::ReadWrite}};
        PlannedTask rd;
        rd.task = 11;
        rd.node = 1;
        rd.operands = {{3, Access::Read}};
        r.tasks = {w, rd};
        ExecutabilityTable run;
        run.add(10, {0});
        run.add(11, {1});
        std::vector<SeqTask> seq{{10, {{3, Access::ReadWrite}}},
            {11, {{3, Access::Read}}}};
        verify(r, seq, run, "write-then-read");
    }

    // Outdated local replica + actual elsewhere: copy into the existing
    // local chunk, never a second binding on that node.
    {
        PlanRequest r;
        r.plan_seq = 5;
        for(ChunkIndex i = 0; i < tiles.n(1); ++i)
        {
            r.entry.push_back({0, ChunkId(i), 1, i, false});
            r.entry.push_back({1, ChunkId(i), 1, i, true});
        }
        PlannedTask t;
        t.task = 12;
        t.node = 0;
        t.operands = {{1, Access::Read}};
        r.tasks.push_back(t);
        ExecutabilityTable run;
        run.add(12, {0});
        std::vector<SeqTask> seq{{12, {{1, Access::Read}}}};
        verify(r, seq, run, "outdated-local");
    }

    // Second batch through the same planner object: per-plan state reset.
    {
        PlanRequest r;
        r.plan_seq = 6;
        r.entry = sink.exit;
        PlannedTask t;
        t.task = 13;
        t.node = 0;
        t.operands = {{3, Access::Read}, {1, Access::Write}};
        r.tasks.push_back(t);
        ExecutabilityTable run;
        run.add(13, {0});
        std::vector<SeqTask> seq{{13, {{3, Access::Read},
            {1, Access::Write}}}};
        verify(r, seq, run, "second-batch");
    }

    if(failures != 0)
    {
        std::fprintf(stderr, "planner_plugin: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("planner_plugin: PASS (checker-valid plan, %zu ops)\n",
        sink.plan.queue.size());
    return 0;
}
