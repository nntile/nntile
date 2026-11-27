/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file tests the
 * timing pass: the three reference fragments reproduced number for
 * number, the concurrency case, and the duration-model-miss rows.
 *
 * @file tests/test_timing.cc
 * The simulator's suite: the model over the node set with the full peer
 * matrix (nine lanes), each fragment a cold-start plan whose every
 * record is asserted against hand-derived arithmetic - peer COPY
 * makespan 240, host-staged chain 2088, host-node TILE_OP 1084; the
 * concurrency case (disjoint TILE_OPs overlap at their own lane
 * clocks); duration-model-miss on a task/node hole and on a channel
 * hole; beside those, the pass's own rules - lane clocks continuous
 * across plan boundaries with cross-plan owed ends, the retire()
 * displacement seam (hygiene, never arithmetic) and its owed-end-miss
 * belt, the run makespan over ALL lanes against a hiding local one,
 * the one-shot simulate_run entry, and the swimlane frames (Full, nine
 * rows with the empties; TouchedSoFar, the monotone union).
 *
 * The fixtures' denomination, stated once: the walk is in MICROSECONDS
 * (D(kernel, g) = 40 us, one host COPY 1024 us, one peer COPY 160 us),
 * and its rates are integral only in that denomination - 15.625 MiB/ms
 * = 16384 bytes/us at the 16 MiB class, while in bytes/ns the same
 * numbers are 16.384 and 104.8576, not integers. The pass and the
 * model move plain uint64 ticks and convert nothing, so the fixture
 * model carries the microsecond numbers as ticks: D(kernel, g) = 40,
 * D(host op) = 20, and the rates RELATIVE to the compile-time class,
 * R(host pair) = chunk_class_bytes() / 1024 and
 * R(g0,g1) = chunk_class_bytes() / 160 - these numbers hold at
 * ANY power-of-two class. Exactness: chunk_class_bytes() = 2^20 * m (m a
 * power of two), so R(host) = 2^10 * m divides exactly and one COPY is
 * exactly 1024; with R(peer) = floor(chunk_class_bytes() / 160), 160 * R
 * <= class < 161 * R holds for every power-of-two class (the deficit r
 * = (2^15 * m) mod 5 leaves 160 * R short of the class by 32r < 161 * R
 * exceeds it by more), so one peer COPY is exactly 160. The two
 * copy_ns() checks below pin the fixture's own arithmetic before any
 * fragment runs.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#include "../src/timing.hh"

// check.hh's shared macro pulls the OLD public header (nnhaul/nnhaul.hh)
// beside the new contract types, and the two disagree on nnhaul::Access
// today - this suite needs no runtime guards, so the CHECK macro stands
// here alone, same shape as check.hh's (and test_binding_table.cc's).
#include "nnhaul/defs.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using namespace nnhaul;

int g_checks_failed = 0;

void check_impl(bool ok, char const *expr, char const *file, int line)
{
    if (!ok)
    {
        std::cerr << "CHECK failed: " << expr << " (" << file << ":" << line
                  << ")\n";
        g_checks_failed += 1;
    }
}

#define CHECK(cond)                                                           \
    check_impl(static_cast<bool>(cond), #cond, __FILE__, __LINE__)

// A loud-throw probe: runs f, true iff it threw std::runtime_error, and
// when needle is given also demands the census-style message carry it.
template <typename F>
bool throws_loud(F &&f, char const *needle = nullptr)
{
    try
    {
        f();
    }
    catch (std::runtime_error const &e)
    {
        if (needle != nullptr &&
            std::string(e.what()).find(needle) == std::string::npos)
        {
            std::cerr << "CHECK failed: the throw message lacks '"
                      << needle << "': " << e.what() << "\n";
            g_checks_failed += 1;
        }
        return true;
    }
    catch (...)
    {
        std::cerr << "CHECK failed: a non-runtime_error exception\n";
        g_checks_failed += 1;
        return true;
    }
    return false;
}

// ---- the shared fixtures (node set, model, the three fragments) ----

// The node set: g0 = 0, g1 = 1, host = 2, the peer matrix fully
// probed - NINE lanes: three execution plus six channel.
NodeInventory make_ii7_inventory()
{
    return NodeInventory(2, {4, 4, 4}, {1, 1, 1, 1});
}

// The task symbols.
TaskId const kKernel = 7;  // D(kernel, g) = 40 (us, as ticks)
TaskId const kHostOp = 9;  // D(host op) = 20
TaskId const kGhost = 11;  // names no model entry - the miss fixture

// The DurationModel key packing (plan.cc's own): (TaskId << 8)
// | NodeId.
std::uint64_t key(TaskId t, NodeId n)
{
    return (static_cast<std::uint64_t>(t) << 8) |
        static_cast<std::uint64_t>(n);
}

// The model, the rates relative to the compile-time class (see the file
// header): every channel of the nine-lane inventory carries a rate -
// the completeness obligation.
DurationModel make_ii7_model()
{
    std::uint64_t const r_host = chunk_class_bytes() / 1024u;
    std::uint64_t const r_peer = chunk_class_bytes() / 160u;

    DurationModel m;
    m.task_node_ns[key(kKernel, 0)] = 40;
    m.task_node_ns[key(kKernel, 1)] = 40;
    m.task_node_ns[key(kHostOp, 2)] = 20;
    m.channel_bytes_per_ns.assign(9, 0); // exec lanes 0..2: not channels
    m.channel_bytes_per_ns[3] = r_peer;  // chan(0,1)
    m.channel_bytes_per_ns[4] = r_host;  // chan(0,2)
    m.channel_bytes_per_ns[5] = r_peer;  // chan(1,0)
    m.channel_bytes_per_ns[6] = r_host;  // chan(1,2)
    m.channel_bytes_per_ns[7] = r_host;  // chan(2,0)
    m.channel_bytes_per_ns[8] = r_host;  // chan(2,1)
    return m;
}

Operand operand(TileId tile, ChunkIndex index, Access access, NodeId node,
    ChunkId chunk)
{
    Operand o;
    o.tile = tile;
    o.index = index;
    o.access = access;
    o.node = node;
    o.chunk = chunk;
    return o;
}

ChunkLoc loc(NodeId node, ChunkId chunk)
{
    return ChunkLoc{node, chunk};
}

// Fragment 1 : peer COPY, one hop. The producer TILE_OP
// on exec(g0) is the fragment's own position 0 (ending t = 40, lane
// clocks at zero, a cold start) - then the COPY over
// chan(g0,g1) owing it, then the reading TILE_OP on exec(g1) owing the
// COPY.
Plan fragment_peer(std::uint64_t seq)
{
    Plan p;
    p.plan_seq = seq;
    p.queue.push_back(make_tile_op(
        kKernel, 0, {operand(1, 0, Access::Write, 0, 0)}));
    p.queue.push_back(make_copy(1, 0, loc(0, 0), loc(1, 0)));
    p.queue.push_back(make_tile_op(
        kKernel, 1, {operand(1, 0, Access::Read, 1, 0)}));
    return p;
}

std::vector<std::vector<OpRef>> owed_peer(std::uint64_t seq)
{
    return {{}, {make_op_ref(seq, 0)}, {make_op_ref(seq, 1)}};
}

// Fragment 2 : the host-staged chain - producer, staging
// LINK, COPY g0->host, COPY host->g1 owing the first COPY's producer of
// h0, residue UNLINK. LINK/UNLINK are never owed (empty entries).
Plan fragment_staged(std::uint64_t seq)
{
    Plan p;
    p.plan_seq = seq;
    p.queue.push_back(make_tile_op(
        kKernel, 0, {operand(2, 0, Access::Write, 0, 0)}));
    p.queue.push_back(make_link(2, 0, 2, 0));
    p.queue.push_back(make_copy(2, 0, loc(0, 0), loc(2, 0)));
    p.queue.push_back(make_copy(2, 0, loc(2, 0), loc(1, 0)));
    p.queue.push_back(make_unlink(2, 0, 2, 0));
    return p;
}

std::vector<std::vector<OpRef>> owed_staged(std::uint64_t seq)
{
    return {{},
        {},
        {make_op_ref(seq, 0)},
        {make_op_ref(seq, 2)},
        {}};
}

// Fragment 3 : host-node TILE_OP reading a g0 chunk -
// producer, staging LINK, COPY g0->host, then the host TILE_OP owing
// the COPY's producer, D 20.
Plan fragment_host_tile(std::uint64_t seq)
{
    Plan p;
    p.plan_seq = seq;
    p.queue.push_back(make_tile_op(
        kKernel, 0, {operand(3, 0, Access::Write, 0, 0)}));
    p.queue.push_back(make_link(3, 0, 2, 1));
    p.queue.push_back(make_copy(3, 0, loc(0, 0), loc(2, 1)));
    p.queue.push_back(make_tile_op(
        kHostOp, 2, {operand(3, 0, Access::Read, 2, 1)}));
    return p;
}

std::vector<std::vector<OpRef>> owed_host_tile(std::uint64_t seq)
{
    return {{}, {}, {make_op_ref(seq, 0)}, {make_op_ref(seq, 2)}};
}

void check_record(SimTraceRecord const &r, std::size_t pos, PlanOpKind kind,
    LaneId lane, std::uint64_t start, std::uint64_t end)
{
    CHECK(r.op_position == pos);
    CHECK(r.kind == kind);
    CHECK(r.lane == lane);
    CHECK(r.start_ns == start);
    CHECK(r.end_ns == end);
}

// ---- the three fragments ----

void test_ii7_fragments()
{
    NodeInventory const inv = make_ii7_inventory();
    DurationModel const model = make_ii7_model();

    // The lane numbering the trace asserts ride on (exec lanes by
    // NodeId, then channel lanes in ascending (a, b) order), and the
    // fixture rates' own arithmetic - 160 and 1024 at ANY compile-time
    // class.
    CHECK(inv.lane_count() == 9);
    CHECK(inv.lane_of_exec(0) == 0);
    CHECK(inv.lane_of_exec(1) == 1);
    CHECK(inv.lane_of_exec(2) == 2);
    CHECK(inv.lane_of_chan(0, 1) == 3);
    CHECK(inv.lane_of_chan(0, 2) == 4);
    CHECK(inv.lane_of_chan(1, 0) == 5);
    CHECK(inv.lane_of_chan(1, 2) == 6);
    CHECK(inv.lane_of_chan(2, 0) == 7);
    CHECK(inv.lane_of_chan(2, 1) == 8);
    CHECK(model.copy_ns(inv.lane_of_chan(0, 1)) == 160u);
    CHECK(model.copy_ns(inv.lane_of_chan(0, 2)) == 1024u);

    // Fragment 1 - peer COPY, makespan 240 : start
    // max(0, 40) = 40, end 200; the reader 200 -> 240.
    {
        TimingPass pass(inv, model);
        PlanTiming const t =
            pass.run_plan(fragment_peer(0), owed_peer(0));
        CHECK(t.local_makespan_ns == 240u);
        CHECK(t.records.size() == 3u);
        check_record(t.records[0], 0, PlanOpKind::TileOp, 0, 0, 40);
        check_record(t.records[1], 1, PlanOpKind::Copy, 3, 40, 200);
        check_record(t.records[2], 2, PlanOpKind::TileOp, 1, 200, 240);
        CHECK(pass.run_makespan() == 240u);
        CHECK(pass.trace().size() == 3u);
    }

    // Fragment 2 - host-staged chain, makespan 2088 :
    // 40 -> 1064 -> 2088 on the two channel lanes, the LINK/UNLINK
    // absent from the trace (no lane, no times).
    {
        TimingPass pass(inv, model);
        PlanTiming const t =
            pass.run_plan(fragment_staged(0), owed_staged(0));
        CHECK(t.local_makespan_ns == 2088u);
        CHECK(t.records.size() == 3u);
        check_record(t.records[0], 0, PlanOpKind::TileOp, 0, 0, 40);
        check_record(t.records[1], 2, PlanOpKind::Copy, 4, 40, 1064);
        check_record(t.records[2], 3, PlanOpKind::Copy, 8, 1064, 2088);
        CHECK(pass.run_makespan() == 2088u);
    }

    // Fragment 3 - host-node TILE_OP, makespan 1084 : the
    // COPY 40 -> 1064, the host op 1064 -> 1084 on exec(host).
    {
        TimingPass pass(inv, model);
        PlanTiming const t =
            pass.run_plan(fragment_host_tile(0), owed_host_tile(0));
        CHECK(t.local_makespan_ns == 1084u);
        CHECK(t.records.size() == 3u);
        check_record(t.records[0], 0, PlanOpKind::TileOp, 0, 0, 40);
        check_record(t.records[1], 2, PlanOpKind::Copy, 4, 40, 1064);
        check_record(t.records[2], 3, PlanOpKind::TileOp, 2, 1064, 1084);
        CHECK(pass.run_makespan() == 1084u);
    }
}

// ---- the concurrency case ----

void test_concurrency()
{
    NodeInventory const inv = make_ii7_inventory();
    DurationModel const model = make_ii7_model();

    // Two disjoint TILE_OPs - no chunk in common, no owed edges - both
    // start at their OWN lane clocks (zero at cold start) and overlap
    // whole against whole despite the queue order; the third op, back
    // on exec(g0), serializes against its lane clock at 40.
    Plan p;
    p.plan_seq = 0;
    p.queue.push_back(make_tile_op(
        kKernel, 0, {operand(4, 0, Access::Write, 0, 0)}));
    p.queue.push_back(make_tile_op(
        kKernel, 1, {operand(5, 0, Access::Write, 1, 0)}));
    p.queue.push_back(make_tile_op(
        kKernel, 0, {operand(4, 0, Access::Read, 0, 0)}));

    TimingPass pass(inv, model);
    PlanTiming const t = pass.run_plan(p, {});
    CHECK(t.records.size() == 3u);
    check_record(t.records[0], 0, PlanOpKind::TileOp, 0, 0, 40);
    check_record(t.records[1], 1, PlanOpKind::TileOp, 1, 0, 40);
    check_record(t.records[2], 2, PlanOpKind::TileOp, 0, 40, 80);
    CHECK(t.local_makespan_ns == 80u);
    CHECK(pass.run_makespan() == 80u);
}

// ---- duration-model-miss on a hole ----

void test_duration_model_miss()
{
    NodeInventory const inv = make_ii7_inventory();
    DurationModel const model = make_ii7_model();

    // A task/node pair the model lacks: D(kGhost, g0) carries no
    // entry - the census names duration-model-miss and the pair.
    {
        Plan p;
        p.plan_seq = 0;
        p.queue.push_back(make_tile_op(kGhost, 0, {}));
        TimingPass pass(inv, model);
        CHECK(throws_loud([&] { pass.run_plan(p, {}); },
            "duration-model-miss"));
        CHECK(throws_loud([&] { pass.run_plan(p, {}); },
            "D(task 11, node 0)"));
    }

    // A channel the model lacks: the same model with chan(0,1)'s rate
    // removed - a COPY over the peered pair censuses with the lane's
    // name.
    {
        DurationModel holed = make_ii7_model();
        holed.channel_bytes_per_ns[inv.lane_of_chan(0, 1)] = 0;
        Plan p;
        p.plan_seq = 0;
        p.queue.push_back(make_copy(1, 0, loc(0, 0), loc(1, 0)));
        TimingPass pass(inv, holed);
        CHECK(throws_loud([&] { pass.run_plan(p, {}); },
            "duration-model-miss"));
        CHECK(throws_loud([&] { pass.run_plan(p, {}); }, "chan(0,1)"));
    }

    // The belt behind the check: a COPY over a channel the INVENTORY
    // lacks has no lane at all - never silently skipped as a
    // LINK/UNLINK would be; the checker's own row name, loud here too.
    {
        NodeInventory const no_peer(2, {4, 4, 4}, {0, 0, 0, 0});
        CHECK(no_peer.lane_count() == 7);
        Plan p;
        p.plan_seq = 0;
        p.queue.push_back(make_copy(1, 0, loc(0, 0), loc(1, 0)));
        TimingPass pass(no_peer, make_ii7_model());
        CHECK(throws_loud(
            [&] { pass.run_plan(p, {}); }, "copy-no-channel"));
    }
}

// ---- the pass's own rules: continuity, the seam, the run makespan ----

void test_continuity_and_seam()
{
    NodeInventory const inv = make_ii7_inventory();
    DurationModel const model = make_ii7_model();

    TimingPass pass(inv, model);

    // Plan 0: the peer fragment - the run's clocks end at exec(g0) 40,
    // exec(g1) 240, chan(0,1) 200.
    PlanTiming const t0 = pass.run_plan(fragment_peer(0), owed_peer(0));
    CHECK(t0.local_makespan_ns == 240u);
    CHECK(pass.run_makespan() == 240u);

    // Retirement by displacement, the seam: every fact still names its
    // op, so the full live set drops nothing; a live ref that was
    // never timed is vacuous, not a fault.
    pass.retire({make_op_ref(0, 0), make_op_ref(0, 1), make_op_ref(0, 2),
        make_op_ref(99, 0)});

    // Plan 1: a TILE_OP on exec(g1) owing plan 0's COPY - a cross-plan
    // owed end (200) under a lane clock that has moved on to 240: the
    // clock wins, 240 -> 280. Lane clocks are CONTINUOUS across plan
    // boundaries , the owed ends come from the run-scoped
    // registry.
    Plan p1;
    p1.plan_seq = 1;
    p1.queue.push_back(
        make_tile_op(kKernel, 1, {operand(1, 0, Access::Read, 1, 0)}));
    PlanTiming const t1 = pass.run_plan(p1, {{make_op_ref(0, 1)}});
    CHECK(t1.records.size() == 1u);
    check_record(t1.records[0], 0, PlanOpKind::TileOp, 1, 240, 280);
    CHECK(t1.local_makespan_ns == 280u);
    CHECK(pass.run_makespan() == 280u);

    // Displacement now drops the producer and the reader of plan 0 -
    // no fact names them any more; the arithmetic is unchanged either
    // way (a displaced fact is never named again).
    pass.retire({make_op_ref(0, 1), make_op_ref(1, 0)});

    // Plan 2: an op on exec(g0), whose clock has stood at 40 since
    // plan 0 - local makespan 80 while the RUN makespan stays 280: the
    // unowed tail on exec(g1) outlives the successor plan, exactly
    // what a per-plan number would hide .
    Plan p2;
    p2.plan_seq = 2;
    p2.queue.push_back(
        make_tile_op(kKernel, 0, {operand(1, 0, Access::Read, 0, 0)}));
    PlanTiming const t2 = pass.run_plan(p2, {});
    check_record(t2.records[0], 0, PlanOpKind::TileOp, 0, 40, 80);
    CHECK(t2.local_makespan_ns == 80u);
    CHECK(pass.run_makespan() == 280u);
    CHECK(pass.trace().size() == 5u); // 3 + 1 + 1, run-wide

    // The belt: a caller defect handing an owed ref that displacement
    // retired - the registry-miss analog, loud, never serviced.
    Plan p3;
    p3.plan_seq = 3;
    p3.queue.push_back(
        make_tile_op(kKernel, 1, {operand(1, 0, Access::Read, 1, 0)}));
    CHECK(throws_loud([&] { pass.run_plan(p3, {{make_op_ref(0, 2)}}); },
        "owed-end-miss"));

    // And the issue-order discipline: a plan_seq that does not follow.
    CHECK(throws_loud([&] { pass.run_plan(fragment_peer(2), owed_peer(2)); },
        "run-queue order"));
}

// ---- the one-shot entry and the swimlane frames ----

void test_simulate_run_and_swimlanes()
{
    NodeInventory const inv = make_ii7_inventory();
    DurationModel const model = make_ii7_model();

    // The one-shot entry: fragment 3 then one plan touching only
    // exec(g1) - the second plan's local makespan is 40 while the RUN
    // makespan stays 1084 (the exec(host) tail; untouched-lane
    // arithmetic by inclusion). The retirement callback runs after
    // each plan; retiring everything changes not one number.
    Plan tail;
    tail.plan_seq = 1;
    tail.queue.push_back(
        make_tile_op(kKernel, 1, {operand(3, 0, Access::Read, 1, 0)}));

    std::size_t retire_calls = 0;
    RunTiming const run = simulate_run(
        {fragment_host_tile(0), tail}, inv, model,
        {owed_host_tile(0)}, // plan 1's view missing: the all-empty view
        [&](std::size_t) {
            retire_calls += 1;
            return std::vector<OpRef>{};
        });
    CHECK(run.plans.size() == 2u);
    CHECK(run.plans[0].local_makespan_ns == 1084u);
    CHECK(run.plans[1].local_makespan_ns == 40u);
    CHECK(run.plans[1].records.size() == 1u);
    check_record(run.plans[1].records[0], 0, PlanOpKind::TileOp, 1, 0, 40);
    CHECK(run.run_makespan_ns == 1084u);
    CHECK(run.trace.size() == 4u);
    CHECK(retire_calls == 2u);

    // The swimlanes over fragment 3 alone: the FULL frame is the
    // default - nine rows, one per lane of the inventory, an empty
    // lane itself the finding, rendered "(idle)"; lanes named
    // exec(N)/chan(A,B), never vendors.
    TimingPass pass(inv, model);
    pass.run_plan(fragment_host_tile(0), owed_host_tile(0));

    SimSwimlanes const full = pass.swimlanes(LaneFrame::Full);
    CHECK(full.frame == LaneFrame::Full);
    CHECK(full.lanes.size() == 9u);
    CHECK(full.lanes[2].lane == 2);
    CHECK(full.lanes[2].segments.size() == 1u);
    CHECK(full.lanes[2].segments[0].op_position == 3u);
    CHECK(full.lanes[1].segments.empty());

    std::string const full_text = pass.render_swimlanes(); // Full: default
    CHECK(full_text ==
        "exec(0)  [p0 tile_op 0..40]\n"
        "exec(1)  (idle)\n"
        "exec(2)  [p3 tile_op 1064..1084]\n"
        "chan(0,1)  (idle)\n"
        "chan(0,2)  [p2 copy 40..1064]\n"
        "chan(1,0)  (idle)\n"
        "chan(1,2)  (idle)\n"
        "chan(2,0)  (idle)\n"
        "chan(2,1)  (idle)\n");
    CHECK(full_text.find("cuda") == std::string::npos);
    CHECK(full_text.find("stream") == std::string::npos);

    // TouchedSoFar: the concession - the monotone union since cold
    // start, never a per-plan subset: three lanes here, and four after
    // a plan that touches only exec(g1).
    std::string const touched_text =
        pass.render_swimlanes(LaneFrame::TouchedSoFar);
    CHECK(touched_text ==
        "exec(0)  [p0 tile_op 0..40]\n"
        "exec(2)  [p3 tile_op 1064..1084]\n"
        "chan(0,2)  [p2 copy 40..1064]\n");

    pass.run_plan(tail, {});
    SimSwimlanes const grown = pass.swimlanes(LaneFrame::TouchedSoFar);
    CHECK(grown.lanes.size() == 4u); // the union grew; plan 1 touched one
    CHECK(grown.lanes[1].lane == 1);
    CHECK(grown.lanes[1].segments.size() == 1u);
    CHECK(pass.swimlanes(LaneFrame::Full).lanes.size() == 9u);

    // The free render agrees with the member render over the record.
    SimSwimlanes const rec = pass.swimlanes(LaneFrame::TouchedSoFar);
    CHECK(render_swimlanes(rec, inv) == pass.render_swimlanes(
        LaneFrame::TouchedSoFar));
}

// A lane is TOUCHED by a zero-duration op though its clock never
// moved - the frame is the union of ops, not of clock motion.
void test_zero_duration_touch()
{
    NodeInventory const inv = make_ii7_inventory();
    DurationModel model = make_ii7_model();
    model.task_node_ns[key(12, 0)] = 0;

    Plan p;
    p.plan_seq = 0;
    p.queue.push_back(make_tile_op(12, 0, {}));

    TimingPass pass(inv, model);
    PlanTiming const t = pass.run_plan(p, {});
    check_record(t.records[0], 0, PlanOpKind::TileOp, 0, 0, 0);
    CHECK(pass.run_makespan() == 0u);
    SimSwimlanes const touched = pass.swimlanes(LaneFrame::TouchedSoFar);
    CHECK(touched.lanes.size() == 1u);
    CHECK(touched.lanes[0].lane == 0);
}

// An empty queue is legal arithmetic: local makespan zero, no records,
// no clock moved.
void test_empty_plan()
{
    NodeInventory const inv = make_ii7_inventory();
    DurationModel const model = make_ii7_model();

    TimingPass pass(inv, model);
    Plan empty;
    empty.plan_seq = 5;
    PlanTiming const t = pass.run_plan(empty, {});
    CHECK(t.local_makespan_ns == 0u);
    CHECK(t.records.empty());
    CHECK(pass.run_makespan() == 0u);
    CHECK(pass.trace().empty());

    Plan next;
    next.plan_seq = 6;
    next.queue.push_back(
        make_tile_op(kKernel, 0, {operand(1, 0, Access::Read, 0, 0)}));
    PlanTiming const t2 = pass.run_plan(next, {});
    check_record(t2.records[0], 0, PlanOpKind::TileOp, 0, 0, 40);
}

} // namespace

int main()
{
    test_ii7_fragments();
    test_concurrency();
    test_duration_model_miss();
    test_continuity_and_seam();
    test_simulate_run_and_swimlanes();
    test_zero_duration_touch();
    test_empty_plan();

    if (g_checks_failed != 0)
    {
        std::cerr << g_checks_failed << " check(s) failed\n";
    }
    return g_checks_failed == 0 ? 0 : 1;
}
