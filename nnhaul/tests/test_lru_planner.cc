/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 * NNHaul is a task-based dynamic execution engine. This file tests the
 * LRU lowering planner: the state read as prefix folds, the four emitted
 * kinds, and the pool-exhaustion census.
 *
 * @file tests/test_lru_planner.cc
 * The lowering walk's suite: the happy path over a 1-GPU+host
 * inventory lowering to LINK/COPY/TILE_OP with the exact predicted
 * exit state, the product validated by the checker passing; the
 * host-staged chain when the peer channel is absent, in its usual form
 * with the residue UNLINK present; eviction under pressure — the LRU
 * victim order by
 * last planned touch verified by position, the data-survival rescue,
 * and the later re-read refetching via COPY; the pool-exhaustion
 * census naming node + need, at the read, with the fold rendered and
 * the ops prefix; the rescue-uncoverable census; replica reads sharing
 * one binding with the write's outdating verified; minimality — only
 * the four kinds ever appear; the deterministic-order pin — the same
 * request produces an identical plan twice; and the loud caller-defect
 * surface. Every product runs through the checker (validate = true).
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#include "../src/lru_planner.hh"

#include "nnhaul/defs.h"

// check.hh's shared macro pulls the OLD public header (nnhaul/nnhaul.hh)
// beside the new contract types, and the two disagree on nnhaul::Access
// today — this suite uses the standalone
// CHECK macro, same shape as check.hh's, until the header piece lands.
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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

// A loud-throw probe: runs f, true iff it threw std::runtime_error,
// and when needle is given also demands the message carry it.
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

// The census catch: runs f, and when PlanCensusError fires, hands the
// census back. `threw` false means the walk shipped a plan it must
// have censused.
struct CaughtCensus
{
    bool threw = false;
    Census census;
};

template <typename F>
CaughtCensus catch_census(F &&f)
{
    CaughtCensus out;
    try
    {
        f();
    }
    catch (PlanCensusError const &e)
    {
        out.threw = true;
        out.census = e.census;
    }
    catch (std::runtime_error const &e)
    {
        std::cerr << "CHECK failed: a plain throw instead of the "
                     "census: "
                  << e.what() << "\n";
        g_checks_failed += 1;
        out.threw = true;
    }
    return out;
}

// ---- fixtures ----

// Tile ids shared by the cases.
constexpr TileId kA = 0;
constexpr TileId kB = 1;
constexpr TileId kC = 2;
constexpr TileId kX = 0;
constexpr TileId kZ = 2;

EntryBinding eb(NodeId node, ChunkId chunk, TileId tile, ChunkIndex index,
    bool actual)
{
    EntryBinding b;
    b.node = node;
    b.chunk = chunk;
    b.tile = tile;
    b.index = index;
    b.actual = actual;
    return b;
}

TileUse use(TileId tile, Access access)
{
    return TileUse{tile, access};
}

PlannedTask task(TaskId id, NodeId node, std::vector<TileUse> operands)
{
    PlannedTask t;
    t.task = id;
    t.node = node;
    t.operands = std::move(operands);
    return t;
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

// A one-device-plus-host inventory (the classic reduction) with the
// given pool cardinalities; no peer bits exist to
// probe at k = 1.
NodeInventory one_gpu(std::size_t device_pool, std::size_t host_pool)
{
    return NodeInventory(1, std::vector<std::size_t>{device_pool, host_pool},
        std::vector<std::uint8_t>{0});
}

// The capped variant: the pools start at their full sizes
// with LIMITS equal to them — the fixed-pool semantics the exhaustion
// census tests need (no growth possible: the limit is the pool).
NodeInventory one_gpu_capped(std::size_t device_pool,
    std::size_t host_pool)
{
    return NodeInventory(1,
        std::vector<std::size_t>{device_pool, host_pool},
        std::vector<std::uint8_t>{0},
        std::vector<std::size_t>{device_pool, host_pool});
}

// The canonical deterministic text render — the byte-identity probe
// for the deterministic-order pin.
std::string dump(PlannedBatch const &b)
{
    auto const num = [](std::uint64_t v) { return std::to_string(v); };
    std::string s;
    s += "plan_seq=" + num(b.plan.plan_seq) + "\n";
    s += "entry:";
    for (EntryBinding const &e : b.plan.entry)
    {
        s += " (" + num(e.node) + "," + num(e.chunk) + ")->" +
            num(e.tile) + "/" + num(e.index) + "/" +
            (e.actual ? "A" : "O");
    }
    s += "\n";
    for (PlanOp const &op : b.plan.queue)
    {
        switch (op.kind)
        {
        case PlanOpKind::Link:
            s += "LINK t" + num(op.u.link.tile) + "/" + num(op.u.link.index) +
                " @" + num(op.u.link.node) + "," + num(op.u.link.chunk);
            break;
        case PlanOpKind::Unlink:
            s += "UNLINK t" + num(op.u.unlink.tile) + "/" +
                num(op.u.unlink.index) + " @" + num(op.u.unlink.node) + "," +
                num(op.u.unlink.chunk);
            break;
        case PlanOpKind::Copy:
            s += "COPY t" + num(op.u.copy.tile) + "/" + num(op.u.copy.index) +
                " " + num(op.u.copy.src.node) + "," +
                num(op.u.copy.src.chunk) + "->" + num(op.u.copy.dst.node) +
                "," + num(op.u.copy.dst.chunk);
            break;
        case PlanOpKind::TileOp:
            s += "TILE_OP " + num(op.u.tile_op.task) + " @" +
                num(op.u.tile_op.node) + " ops:";
            for (Operand const &e : op.operands)
            {
                s += " " + num(e.tile) + "/" + num(e.index) + "/" +
                    num(static_cast<unsigned>(e.access)) + "@" +
                    num(e.node) + "," + num(e.chunk);
            }
            break;
        }
        s += "\n";
    }
    s += "exit:";
    for (EntryBinding const &e : b.exit)
    {
        s += " (" + num(e.node) + "," + num(e.chunk) + ")->" + num(e.tile) +
            "/" + num(e.index) + "/" + (e.actual ? "A" : "O");
    }
    s += "\n";
    return s;
}

// Queue equality against a hand-built expectation, op for op.
bool queue_is(Plan const &plan, std::vector<PlanOp> const &expected)
{
    if (plan.queue.size() != expected.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        if (!(plan.queue[i] == expected[i]))
        {
            return false;
        }
    }
    return true;
}

// ---- 1. the happy path: LINK/COPY/TILE_OP and the exact exit ----

// One device (node 0, pool 3) plus the host (node 1, pool 4); tile A
// of one chunk and tile B of TWO chunks (the whole-tile pin's visible
// form); the entry holds A actual on the host. One task reads A and
// writes B on the device: LINK+COPY for A's operand (the host is the
// COPY source), two bare LINKs for B's chunks (a Write needs no data),
// ONE whole-tile TILE_OP with three operand entries.
void test_happy_path()
{
    NodeInventory const inv = one_gpu(3, 4);
    TileTable tiles;
    tiles.add(kA, chunk_class_bytes());              // n = 1
    tiles.add(kB, 2 * chunk_class_bytes());          // n = 2

    PlanRequest req;
    req.plan_seq = 7;
    req.entry = {eb(1, 0, kA, 0, true)};
    req.tasks = {task(7, 0, {use(kA, Access::Read), use(kB, Access::Write)})};

    LruPlanner planner;
    PlannedBatch const out = planner.plan(req, inv, tiles, true);

    std::vector<PlanOp> const expected = {
        make_link(kA, 0, 0, 0),
        make_copy(kA, 0, ChunkLoc{1, 0}, ChunkLoc{0, 0}),
        make_link(kB, 0, 0, 1),
        make_link(kB, 1, 0, 2),
        make_tile_op(7, 0,
            {{operand(kA, 0, Access::Read, 0, 0),
                operand(kB, 0, Access::Write, 0, 1),
                operand(kB, 1, Access::Write, 0, 2)}}),
    };
    CHECK(queue_is(out.plan, expected));

    std::vector<EntryBinding> const expected_exit = {
        eb(0, 0, kA, 0, true), // COPY's destination, actual
        eb(0, 1, kB, 0, true), // written by the TILE_OP
        eb(0, 2, kB, 1, true), // written by the TILE_OP
        eb(1, 0, kA, 0, true), // the entry replica, read-only untouched
    };
    CHECK(out.exit == expected_exit);
    CHECK(out.plan.entry == req.entry); // sorted by (node, chunk) as given
    CHECK(out.plan.plan_seq == 7);

    // The conversion surface: the request's tasks ARE the checker's
    // task sequence .
    std::vector<SeqTask> const seq = to_seq_tasks(req.tasks);
    CHECK(seq.size() == 1);
    CHECK(seq[0].task == 7);
    CHECK(seq[0].operands.size() == 2);
    CHECK(seq[0].operands[0].tile == kA &&
        seq[0].operands[0].access == Access::Read);
    CHECK(seq[0].operands[1].tile == kB &&
        seq[0].operands[1].access == Access::Write);
}

// ---- 2. the host-staged chain when the peer channel is absent ----

// Two GPUs (nodes 0, 1) plus the host (node 2), NO peer bits: a g0 ->
// g1 move is staged through the host in the usual two-hop form — the
// destination LINK, then LINK(host) + COPY(g0->host) +
// COPY(host->g1) + UNLINK(host), the residue UNLINK present.
void test_host_staged_chain()
{
    NodeInventory const inv(2, std::vector<std::size_t>{2, 2, 4},
        std::vector<std::uint8_t>{0, 0, 0, 0});

    TileTable tiles;
    tiles.add(kX, chunk_class_bytes());

    PlanRequest req;
    req.plan_seq = 1;
    req.entry = {eb(0, 0, kX, 0, true)}; // X actual on g0
    req.tasks = {task(3, 1, {use(kX, Access::Read)})}; // ...read on g1

    LruPlanner planner;
    PlannedBatch const out = planner.plan(req, inv, tiles, true);

    std::vector<PlanOp> const expected = {
        make_link(kX, 0, 1, 0),  // g1's own operand binding
        make_link(kX, 0, 2, 0),  // the staging hop on the host
        make_copy(kX, 0, ChunkLoc{0, 0}, ChunkLoc{2, 0}),
        make_copy(kX, 0, ChunkLoc{2, 0}, ChunkLoc{1, 0}),
        make_unlink(kX, 0, 2, 0), // the staging residue
        make_tile_op(3, 1, {{operand(kX, 0, Access::Read, 1, 0)}}),
    };
    CHECK(queue_is(out.plan, expected));

    // The residue is gone from the exit: only g0's original and g1's
    // operand binding remain, both actual.
    std::vector<EntryBinding> const expected_exit = {
        eb(0, 0, kX, 0, true),
        eb(1, 0, kX, 0, true),
    };
    CHECK(out.exit == expected_exit);
}

// ---- 3. eviction under pressure: LRU order, rescue, refetch ----

// Device pool of TWO, both chunks linked at entry (A and B, actual).
// Task 0 reads A (A's touch makes B the older binding); task 1 writes
// C — the free list is empty, the LRU victim is B (never touched this
// build — A's last planned touch is task 0's TILE_OP), and B is read
// again by task 2, so the data-survival rule rescues it to the host
// BEFORE the UNLINK; task 2's re-read then refetches via COPY. A, its
// last read behind the walk, dies bare (the dying-victim retirement rule).
void test_eviction_under_pressure()
{
    // The device pool is CAPPED at its full size (2): the
    // pressure this test builds needs eviction, not growth.
    NodeInventory const inv = one_gpu_capped(2, 4);
    TileTable tiles;
    tiles.add(kA, chunk_class_bytes());
    tiles.add(kB, chunk_class_bytes());
    tiles.add(kC, chunk_class_bytes());

    PlanRequest req;
    req.plan_seq = 2;
    req.entry = {eb(0, 0, kA, 0, true), eb(0, 1, kB, 0, true)};
    req.tasks = {task(10, 0, {use(kA, Access::Read)}),
        task(11, 0, {use(kC, Access::Write)}),
        task(12, 0, {use(kB, Access::Read)})};

    LruPlanner planner;
    PlannedBatch const out = planner.plan(req, inv, tiles, true);

    // CROSS-BATCH SURVIVAL (the engine-owned rule): A is LIVE (not in
    // the request's dead set) even though no reader of A remains in
    // THIS batch — its last actual replica is RESCUED to the host,
    // never unlinked bare. A live tile's readers may live in later
    // batches (a production fault was the batch-scoped rule).
    std::vector<PlanOp> const expected = {
        // task 0: the replica read of A, no ops of its own
        make_tile_op(10, 0, {{operand(kA, 0, Access::Read, 0, 0)}}),
        // task 1: B evicted (the LRU front — untouched this build,
        // older than A's task-0 touch) but rescued to the host first
        make_link(kB, 0, 1, 0),
        make_copy(kB, 0, ChunkLoc{0, 1}, ChunkLoc{1, 0}),
        make_unlink(kB, 0, 0, 1),
        // ...then C's fresh binding takes the freed chunk
        make_link(kC, 0, 0, 1),
        make_tile_op(11, 0, {{operand(kC, 0, Access::Write, 0, 1)}}),
        // task 2: A evicted and RESCUED (live, no remaining in-batch
        // reader — the rescue is the engine's cross-batch survival),
        // then B refetches from its host rescue replica
        make_link(kA, 0, 1, 1),
        make_copy(kA, 0, ChunkLoc{0, 0}, ChunkLoc{1, 1}),
        make_unlink(kA, 0, 0, 0),
        make_link(kB, 0, 0, 0),
        make_copy(kB, 0, ChunkLoc{1, 0}, ChunkLoc{0, 0}),
        make_tile_op(12, 0, {{operand(kB, 0, Access::Read, 0, 0)}}),
    };
    CHECK(queue_is(out.plan, expected));

    // The victim order verified BY POSITION: the first eviction in the
    // queue is B's UNLINK (position 3), not A's (position 8) — B's
    // last planned touch predates A's; and every actual victim's
    // UNLINK is preceded by its rescue COPY (positions 2 and 7).
    CHECK(out.plan.queue[3].kind == PlanOpKind::Unlink &&
        out.plan.queue[3].u.unlink.tile == kB);
    CHECK(out.plan.queue[8].kind == PlanOpKind::Unlink &&
        out.plan.queue[8].u.unlink.tile == kA);

    std::vector<EntryBinding> const expected_exit = {
        eb(0, 0, kB, 0, true), // refetched and read
        eb(0, 1, kC, 0, true), // written
        eb(1, 0, kB, 0, true), // B's rescue replica, still actual
        eb(1, 1, kA, 0, true), // A's rescue replica: the bytes survived
    };
    CHECK(out.exit == expected_exit);
}

// ---- 3b. the dead tile's bare eviction (the survival rule's other
//          half: the engine's retirement set is the ONLY tile whose
//          bytes may die at an eviction) ----

void test_dead_tile_evicts_bare()
{
    NodeInventory const inv = one_gpu(2, 4);
    TileTable tiles;
    tiles.add(kA, chunk_class_bytes());
    tiles.add(kB, chunk_class_bytes());
    tiles.add(kC, chunk_class_bytes());

    // The same shape as the pressure case, but A rides the batch's
    // retirement (the request's dead set): its eviction needs no
    // rescue — the trailing UNLINKs drop its bindings anyway.
    PlanRequest req;
    req.plan_seq = 3;
    req.entry = {eb(0, 0, kA, 0, true), eb(0, 1, kB, 0, true)};
    req.tasks = {task(10, 0, {use(kA, Access::Read)}),
        task(11, 0, {use(kC, Access::Write)}),
        task(12, 0, {use(kB, Access::Read)})};
    req.dead_tiles = {kA};

    LruPlanner planner;
    PlannedBatch const out = planner.plan(req, inv, tiles, true);

    // A's eviction is BARE: no COPY of A exists anywhere in the queue.
    bool a_copy = false;
    for (PlanOp const &op : out.plan.queue)
    {
        if (op.kind == PlanOpKind::Copy && op.u.copy.tile == kA)
        {
            a_copy = true;
        }
    }
    CHECK(!a_copy);
    bool a_bare_unlink = false;
    for (PlanOp const &op : out.plan.queue)
    {
        if (op.kind == PlanOpKind::Unlink && op.u.unlink.tile == kA)
        {
            a_bare_unlink = true;
        }
    }
    CHECK(a_bare_unlink);

    // The exit holds no binding of A at all: the dead tile's bytes
    // are gone, legally (its retirement rides this batch).
    for (EntryBinding const &b : out.exit)
    {
        CHECK(b.tile != kA);
    }
}

// ---- 4. the pool-exhaustion census ----

// Device pool of ONE, linked to A at entry; the task reads A and
// writes B — B's binding need finds the free list empty and the only
// linked chunk is the current task's own operand (the working-set
// guard): pool-exhaustion, censused at the state read, naming the
// node and the need.
void test_pool_exhaustion_census()
{
    NodeInventory const inv = one_gpu_capped(1, 2);
    TileTable tiles;
    tiles.add(kA, chunk_class_bytes());
    tiles.add(kB, chunk_class_bytes());

    PlanRequest req;
    req.plan_seq = 3;
    req.entry = {eb(0, 0, kA, 0, true)};
    req.tasks = {task(5, 0, {use(kA, Access::Read), use(kB, Access::Write)})};

    LruPlanner planner;
    CaughtCensus const caught = catch_census(
        [&] { planner.plan(req, inv, tiles, true); });
    CHECK(caught.threw);
    CHECK(caught.census.fault == "pool-exhaustion");
    CHECK(!caught.census.op.has_value()); // the read faults, not an op

    // The census names the node and the need (the position column at
    // plan time: which task / tile chunk).
    CHECK(caught.census.inputs.find("node=0") != std::string::npos);
    CHECK(caught.census.inputs.find("tid=5") != std::string::npos);
    CHECK(caught.census.inputs.find("tile=" + std::to_string(kB)) !=
        std::string::npos);
    CHECK(caught.census.inputs.find("chunk=0") != std::string::npos);

    // The fold snapshot, per node with the free list beside it: node
    // 0 holds A linked with an EMPTY free list; the host segment is
    // rendered too.
    CHECK(caught.census.fold.size() == 2);
    CHECK(caught.census.fold[0].node == 0);
    CHECK(caught.census.fold[0].pool == 1);
    CHECK(caught.census.fold[0].bindings.size() == 1);
    CHECK(caught.census.fold[0].bindings[0].tile == kA);
    CHECK(caught.census.fold[0].bindings[0].actual);
    CHECK(caught.census.fold[0].free_list.empty());
    CHECK(caught.census.fold[1].node == 1);
    CHECK(caught.census.fold[1].bindings.empty());

    // The entry state and the ops prefix (empty here — the need struck
    // before any op) travel with the census.
    CHECK(caught.census.entry == req.entry);
    CHECK(caught.census.prefix.empty());
}

// The rescue's own uncoverable case: the victim is the LAST actual
// replica, a later task reads it, and no channel-bearing node can
// land the copy (the host pool of one is linked to Z, no outdated
// binding anywhere) — pool-exhaustion again, the census naming the
// evicting node and the need, the blocker rendered.
void test_rescue_uncoverable_census()
{
    NodeInventory const inv = one_gpu_capped(1, 1);
    TileTable tiles;
    tiles.add(kA, chunk_class_bytes());
    tiles.add(kB, chunk_class_bytes());
    tiles.add(kZ, chunk_class_bytes());

    PlanRequest req;
    req.plan_seq = 4;
    req.entry = {eb(0, 0, kA, 0, true), eb(1, 0, kZ, 0, false)};
    req.tasks = {task(5, 0, {use(kB, Access::Write)}),
        task(6, 0, {use(kA, Access::Read)})};

    LruPlanner planner;
    CaughtCensus const caught = catch_census(
        [&] { planner.plan(req, inv, tiles, true); });
    CHECK(caught.threw);
    CHECK(caught.census.fault == "pool-exhaustion");
    CHECK(caught.census.inputs.find("node=0") != std::string::npos);
    CHECK(caught.census.inputs.find("rescue") != std::string::npos);
}

// ---- 5. replica reads: one shared binding, the write outdates ----

void test_replica_reads()
{
    NodeInventory const inv = one_gpu(2, 4);
    TileTable tiles;
    tiles.add(kA, chunk_class_bytes());

    // Write then read: the second task shares the first's binding —
    // no LINK, no COPY between the two TILE_OPs — and the first's
    // write leaves the entry's HOST replica outdated (the outdating).
    {
        PlanRequest req;
        req.plan_seq = 5;
        req.entry = {eb(1, 0, kA, 0, true)};
        req.tasks = {task(1, 0, {use(kA, Access::Write)}),
            task(2, 0, {use(kA, Access::Read)})};

        LruPlanner planner;
        PlannedBatch const out = planner.plan(req, inv, tiles, true);

        std::vector<PlanOp> const expected = {
            make_link(kA, 0, 0, 0),
            make_tile_op(1, 0, {{operand(kA, 0, Access::Write, 0, 0)}}),
            make_tile_op(2, 0, {{operand(kA, 0, Access::Read, 0, 0)}}),
        };
        CHECK(queue_is(out.plan, expected));
        // The shared binding: both TILE_OPs name chunk 0 (above), and
        // nothing was emitted between them.
        CHECK(out.plan.queue[2].operands[0].chunk ==
            out.plan.queue[1].operands[0].chunk);

        std::vector<EntryBinding> const expected_exit = {
            eb(0, 0, kA, 0, true),  // written, actual
            eb(1, 0, kA, 0, false), // outdated by the write
        };
        CHECK(out.exit == expected_exit);
    }

    // Read then read: the entry binding serves both, ZERO ops beside
    // the two TILE_OPs.
    {
        PlanRequest req;
        req.plan_seq = 6;
        req.entry = {eb(0, 0, kA, 0, true)};
        req.tasks = {task(1, 0, {use(kA, Access::Read)}),
            task(2, 0, {use(kA, Access::Read)})};

        LruPlanner planner;
        PlannedBatch const out = planner.plan(req, inv, tiles, true);
        CHECK(out.plan.queue.size() == 2);
        CHECK(out.plan.queue[0].kind == PlanOpKind::TileOp);
        CHECK(out.plan.queue[1].kind == PlanOpKind::TileOp);
        CHECK(out.exit == req.entry);
    }
}

// ---- 6. minimality: only the four kinds, ever ----

void test_minimality()
{
    NodeInventory const inv = one_gpu(2, 4);
    TileTable tiles;
    tiles.add(kA, chunk_class_bytes());
    tiles.add(kB, chunk_class_bytes());

    PlanRequest req;
    req.plan_seq = 8;
    req.entry = {eb(1, 0, kA, 0, true), eb(0, 0, kB, 0, false)};
    req.tasks = {task(1, 0, {use(kA, Access::Read), use(kB, Access::Write)}),
        task(2, 0, {use(kB, Access::ReadWrite)})};

    LruPlanner planner;
    PlannedBatch const out = planner.plan(req, inv, tiles, true);

    CHECK(!out.plan.queue.empty());
    for (PlanOp const &op : out.plan.queue)
    {
        bool const kind_ok = op.kind == PlanOpKind::Link ||
            op.kind == PlanOpKind::Unlink || op.kind == PlanOpKind::Copy ||
            op.kind == PlanOpKind::TileOp;
        CHECK(kind_ok);
        // Only TILE_OP carries operands (each op carries exactly its own
        // fields); the queue carries nothing but kind, payload, and
        // the operand list — structurally guaranteed by PlanOp, the
        // kinds asserted here.
        CHECK(op.kind == PlanOpKind::TileOp || op.operands.empty());
    }
}

// ---- 7. the deterministic-order pin: same request, identical plan ----

void test_determinism()
{
    NodeInventory const inv = one_gpu(3, 4);
    TileTable tiles;
    tiles.add(kA, chunk_class_bytes());
    tiles.add(kB, 2 * chunk_class_bytes());

    PlanRequest req;
    req.plan_seq = 9;
    req.entry = {eb(1, 0, kA, 0, true), eb(0, 0, kB, 1, false)};
    req.tasks = {task(7, 0, {use(kA, Access::Read), use(kB, Access::Write)}),
        task(8, 0, {use(kB, Access::ReadWrite)}),
        task(9, 0, {use(kA, Access::Read), use(kB, Access::Read)})};

    LruPlanner planner;
    PlannedBatch const first = planner.plan(req, inv, tiles, true);
    PlannedBatch const second = planner.plan(req, inv, tiles, true);

    CHECK(first.plan.plan_seq == second.plan.plan_seq);
    CHECK(first.plan.entry == second.plan.entry);
    CHECK(queue_is(second.plan, first.plan.queue));
    CHECK(first.exit == second.exit);
    CHECK(dump(first) == dump(second)); // the byte-identity probe
}

// ---- 8. the loud caller-defect surface (never a census) ----

void test_caller_defects()
{
    NodeInventory const inv = one_gpu(2, 4);
    TileTable tiles;
    tiles.add(kA, chunk_class_bytes());
    tiles.add(kB, chunk_class_bytes());

    LruPlanner planner;

    // A task naming one tile twice — a read-and-written tile is one
    // ReadWrite use .
    CHECK(throws_loud([&] {
        PlanRequest req;
        req.tasks = {task(
            1, 0, {use(kA, Access::Read), use(kA, Access::Write)})};
        planner.plan(req, inv, tiles, true);
    }, "twice"));

    // A tile the table does not carry.
    CHECK(throws_loud([&] {
        PlanRequest req;
        req.tasks = {task(1, 0, {use(9, Access::Read)})};
        planner.plan(req, inv, tiles, true);
    }, "does not carry"));

    // A repeated TaskId in the batch is LEGAL (by the collapse rule: the
    // symbol sequence matches the task sequence one-for-one;
    // the engine's multiplied task symbols map many instances onto one
    // symbol) — the walk plans both, in submit order.
    {
        PlanRequest req;
        req.tasks = {task(1, 0, {use(kA, Access::Write)}),
            task(1, 0, {use(kB, Access::Write)})};
        PlannedBatch out = planner.plan(req, inv, tiles, true);
        CHECK(out.plan.queue.size() >= 2);
        std::size_t tile_ops = 0;
        for (PlanOp const &op : out.plan.queue)
        {
            if (op.kind == PlanOpKind::TileOp)
            {
                CHECK(op.u.tile_op.task == 1);
                tile_ops += 1;
            }
        }
        CHECK(tile_ops == 2);
    }

    // The entry naming one (node, chunk) twice.
    CHECK(throws_loud([&] {
        PlanRequest req;
        req.entry = {eb(0, 0, kA, 0, true), eb(0, 0, kB, 0, false)};
        req.tasks = {task(1, 0, {use(kA, Access::Read)})};
        planner.plan(req, inv, tiles, true);
    }, "twice"));

    // A read of data the predicted world never produced: no actual
    // binding anywhere can feed the COPY — a caller defect, not a
    // plan-time fault (pool-exhaustion alone is a plan-time fault).
    CHECK(throws_loud([&] {
        PlanRequest req;
        req.tasks = {task(1, 0, {use(kA, Access::Read)})};
        planner.plan(req, inv, tiles, true);
    }, "no actual binding"));
}

} // namespace

int main()
{
    test_happy_path();
    test_host_staged_chain();
    test_eviction_under_pressure();
    test_dead_tile_evicts_bare();
    test_pool_exhaustion_census();
    test_rescue_uncoverable_census();
    test_replica_reads();
    test_minimality();
    test_determinism();
    test_caller_defects();

    if (g_checks_failed != 0)
    {
        std::cerr << g_checks_failed << " check(s) failed\n";
    }
    return g_checks_failed == 0 ? 0 : 1;
}
