/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: one in-order CUDA
 * stream executes submitted tasks while a reactive data manager moves
 * bytes between the device and pinned host memory on demand.
 *
 * @file tests/test_dependency.cc
 * The owed derivation's suite : the update tables row
 * by row over a scratch op sequence; the never-reset USER fact across
 * an UNLINK/LINK epoch ;
 * same-lane elision ; the COPY row's destination half — the
 * latest USER of d on each lane != lane(O), WAR and WAW alike; the
 * domination fold — three same-lane
 * debts, one survivor, and the per-consumer mint note ; the
 * mint iff (a never-owed producer mints nothing, a cross-lane owed
 * producer mints exactly once); late records across a plan
 * boundary; registry-miss on a hand-broken wait ; and sweep
 * displacement — entries dropped exactly when no fact names them, an
 * entry still named by a USER fact from an old epoch surviving the
 * sweep even after its plan closed.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "../src/dependency.hh"

// check.hh's shared macro pulls the OLD public header (nnhaul/nnhaul.hh)
// beside the new contract types, and the two disagree on nnhaul::Access
// today — this suite needs no runtime guards, so the CHECK macro stands
// here alone, same shape as check.hh's.
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

// A loud-throw probe: runs f, true iff it threw std::runtime_error,
// and when needle is given also demands the message carry it (the
// fault's name must be greppable).
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

// ---- the fixtures ----

// Two device nodes (0, 1) plus the host (2), pools of 4 chunks each.
// With peer = true the 0<->1 device pair is probed both directions, so
// the run's lanes are exec(0), exec(1), exec(2) then the six channels
// (0,1), (0,2), (1,0), (1,2), (2,0), (2,1) — every lane family exists.
NodeInventory make_inventory(bool peer)
{
    return NodeInventory(2, std::vector<std::size_t>{4, 4, 4},
        peer ? std::vector<std::uint8_t>{0, 1, 1, 0}
             : std::vector<std::uint8_t>{0, 0, 0, 0});
}

Operand opnd(TileId t, ChunkIndex i, Access a, NodeId n, ChunkId c)
{
    Operand o;
    o.tile = t;
    o.index = i;
    o.access = a;
    o.node = n;
    o.chunk = c;
    return o;
}

// ---- the update tables, row by row  ----

void test_update_tables()
{
    NodeInventory const inv = make_inventory(true);
    DependencyState dep(inv);

    // Cold start: no facts anywhere .
    CHECK(dep.producer(ChunkLoc{0, 0}) == kNoRef);
    CHECK(dep.user(ChunkLoc{0, 0}, 0) == kNoRef);
    CHECK(dep.user(ChunkLoc{0, 0}, 1) == kNoRef);

    // TILE_OP row 1: USER(each operand, lane(O)) := O — every operand
    // is a user in ANY mode, the Read included; row 2: PRODUCER(each
    // written operand) := O — ReadWrite writes, Read does not.
    PlanOp const w = make_tile_op(1, 0,
        {opnd(0, 0, Access::ReadWrite, 0, 0),
            opnd(1, 0, Access::Read, 0, 1)});
    OpRef const W = make_op_ref(0, 0);
    dep.after(w, W, lane_of_op(w, inv)); // exec(0)
    CHECK(dep.user(ChunkLoc{0, 0}, 0) == W);      // written operand
    CHECK(dep.user(ChunkLoc{0, 1}, 0) == W);      // Read operand too
    CHECK(dep.producer(ChunkLoc{0, 0}) == W);     // ReadWrite produces
    CHECK(dep.producer(ChunkLoc{0, 1}) == kNoRef); // Read does not
    CHECK(dep.user(ChunkLoc{0, 0}, 1) == kNoRef); // other lanes untouched

    // A TILE_OP on another lane: facts move forward per (chunk, lane),
    // the reader is a user but never a producer.
    PlanOp const r = make_tile_op(2, 1, {opnd(0, 0, Access::Read, 0, 0)});
    OpRef const R = make_op_ref(0, 1);
    dep.after(r, R, lane_of_op(r, inv)); // exec(1)
    CHECK(dep.user(ChunkLoc{0, 0}, 1) == R);
    CHECK(dep.user(ChunkLoc{0, 0}, 0) == W);  // lane 0's fact untouched
    CHECK(dep.producer(ChunkLoc{0, 0}) == W); // a reader is not a writer

    // COPY row: USER(src, lane) := O; PRODUCER(dst) := O; USER(dst,
    // lane) := O — the copy is the source's latest USER and the
    // destination's latest producer AND user; the source's PRODUCER
    // fact is untouched.
    PlanOp const cp = make_copy(0, 0, ChunkLoc{0, 0}, ChunkLoc{1, 0});
    OpRef const C = make_op_ref(0, 2);
    dep.after(cp, C, lane_of_op(cp, inv)); // chan(0, 1)
    CHECK(dep.user(ChunkLoc{0, 0}, lane_of_op(cp, inv)) == C);
    CHECK(dep.producer(ChunkLoc{0, 0}) == W); // the copy does not write src
    CHECK(dep.producer(ChunkLoc{1, 0}) == C);
    CHECK(dep.user(ChunkLoc{1, 0}, lane_of_op(cp, inv)) == C);

    // UNLINK row: the PRODUCER fact dies (the epoch-scoped death), the
    // USER facts live on .
    PlanOp const ul = make_unlink(0, 0, 1, 0);
    dep.after(ul, make_op_ref(0, 3), kNoLane);
    CHECK(dep.producer(ChunkLoc{1, 0}) == kNoRef);
    CHECK(dep.user(ChunkLoc{1, 0}, lane_of_op(cp, inv)) == C);
    CHECK(dep.user(ChunkLoc{0, 0}, 0) == W);
    CHECK(dep.user(ChunkLoc{0, 0}, 1) == R);

    // LINK row: nothing, not even the resurrected PRODUCER.
    PlanOp const lk = make_link(9, 0, 1, 0);
    dep.after(lk, make_op_ref(0, 4), kNoLane);
    CHECK(dep.producer(ChunkLoc{1, 0}) == kNoRef);
    CHECK(dep.user(ChunkLoc{1, 0}, lane_of_op(cp, inv)) == C);

    // The loud programmer-range surface (the belt under the
    // unknown-id rows): a chunk outside its node's pool, a node
    // outside the inventory, a lane outside the run's lane set.
    CHECK(throws_loud([&] { (void)dep.user(ChunkLoc{0, 99}, 0); }));
    CHECK(throws_loud([&] { (void)dep.producer(ChunkLoc{5, 0}); }));
    CHECK(throws_loud([&] { (void)dep.user(ChunkLoc{0, 0}, 9); }));
}

// ---- the never-reset USER across an UNLINK/LINK epoch  ----

void test_user_never_resets()
{
    NodeInventory const inv = make_inventory(true);
    Derivation d(inv);

    Plan plan;
    plan.plan_seq = 0;
    plan.queue.push_back(make_tile_op(2, 1,
        {opnd(0, 0, Access::Read, 0, 0)})); // p0: the in-flight reader
    plan.queue.push_back(make_unlink(0, 0, 0, 0)); // p1: the epoch dies
    plan.queue.push_back(make_link(9, 0, 0, 0));   // p2: the slot relinks
    plan.queue.push_back(make_tile_op(1, 0,
        {opnd(9, 0, Access::Write, 0, 0)})); // p3: the new-epoch writer
    PlanDerivation const r = d.run(plan);

    OpRef const reader = make_op_ref(0, 0);
    OpRef const writer = make_op_ref(0, 3);

    // LINK/UNLINK owe nothing; the reader owed nothing (no producer).
    CHECK(r.owed_set[0].empty());
    CHECK(r.owed_set[1].empty());
    CHECK(r.owed_set[2].empty());

    // The new-epoch writer on another lane consults the OLD reader:
    // the USER fact survived the unlink, the relink, and the epoch
    // boundary — the bytes do not know which epoch they serve.
    CHECK(r.owed_set[3] == std::vector<OpRef>{reader});
    CHECK(r.mints[0]); // the reader's event mints: a cross-lane debt
    CHECK(!r.mints[3]);
    CHECK(r.late_records.empty());

    // The facts as carried: the old reader's USER fact still names it;
    // the writer is the new epoch's producer (and lane 0's user).
    CHECK(d.state().user(ChunkLoc{0, 0}, 1) == reader);
    CHECK(d.state().producer(ChunkLoc{0, 0}) == writer);
    CHECK(d.state().user(ChunkLoc{0, 0}, 0) == writer);
}

// ---- same-lane elision  ----

void test_same_lane_elision()
{
    NodeInventory const inv = make_inventory(true);
    Derivation d(inv);

    Plan plan;
    plan.plan_seq = 0;
    plan.queue.push_back(
        make_tile_op(1, 1, {opnd(0, 0, Access::ReadWrite, 1, 0)})); // p0
    plan.queue.push_back(
        make_tile_op(2, 1, {opnd(0, 0, Access::Read, 1, 0)})); // p1
    plan.queue.push_back(
        make_tile_op(2, 1, {opnd(0, 0, Access::Write, 1, 0)})); // p2
    // p3/p4: two copies over the same channel lane (1 -> host).
    plan.queue.push_back(make_copy(
        0, 0, ChunkLoc{1, 0}, ChunkLoc{2, 0})); // p3
    plan.queue.push_back(make_copy(
        0, 0, ChunkLoc{1, 0}, ChunkLoc{2, 0})); // p4
    PlanDerivation const r = d.run(plan);

    // Refs by position: p0 ReadWrite A, p1 same-lane reader B, p2 the
    // same-lane writer D (kept below), p3 the first copy E, p4 the
    // second copy F.
    OpRef const D = make_op_ref(0, 2);

    // p1: the producer A is on p1's own lane — the lane's in-order
    // semantics carry it; no edge materializes.
    CHECK(r.owed_set[1].empty());

    // p2: the only USER fact of the chunk lives on p2's own lane (A,
    // then B, then D itself forward) — nothing owed from the user row.
    CHECK(r.owed_set[2].empty());

    // p3: the copy owes the chunk's producer D — a CROSS-lane debt
    // (exec(1) vs chan(1,2)), so it does owe exactly that. p3
    // and p4 are consecutive on the chan lane — ONE segment; p3 is
    // the first member, its {D} is a new max on exec(1): KEPT, the
    // wait at p3's own position.
    CHECK(r.owed_set[3] == std::vector<OpRef>{D});

    // p4: the second member of p3's segment — its raw fold (the same
    // {D}, plus the same-lane-elided destination user E) is DOMINATED
    // by p3's kept D on exec(1): p4's work follows p3's work on the
    // one in-order chan stream, which waited D — the edge holds a
    // fortiori, the ref is DROPPED, and the wait walk never happens
    // (transitive elision, made structural at derive time).
    CHECK(r.owed_set[4].empty());

    // The same-lane debt never mints an event (never for a same-lane
    // debt): E is owed by nobody, B and A likewise; D,
    // owed cross-lane by p3's kept row, mints.
    CHECK(r.mints[0] == false);
    CHECK(r.mints[1] == false);
    CHECK(r.mints[3] == false); // E itself: only F's fold saw it, elided
    CHECK(r.mints[2] == true);  // D
    CHECK(r.late_records.empty());
}

// ---- the COPY row's destination half : the latest USER
// of d on each lane != lane(O) ----

void test_copy_dst_users()
{
    NodeInventory const inv = make_inventory(true);
    Derivation d(inv);

    Plan plan;
    plan.plan_seq = 0;
    // T0 on exec(0) WRITES the host chunk (2,0) — the WAW toucher of
    // the copy's future destination.
    plan.queue.push_back(
        make_tile_op(1, 0, {opnd(0, 0, Access::Write, 2, 0)}));
    // Rd on exec(2) reads (2,0) — a WAR toucher on yet another lane
    // (and itself a cross-lane consumer of T0).
    plan.queue.push_back(
        make_tile_op(2, 2, {opnd(0, 0, Access::Read, 2, 0)}));
    // W on exec(1) writes the copy's future source (1,0).
    plan.queue.push_back(
        make_tile_op(1, 1, {opnd(0, 0, Access::Write, 1, 0)}));
    // The COPY (1,0) -> (2,0) over chan(1,2): its fold must name the
    // destination's latest user on EVERY lane != chan(1,2) — T0's WAW
    // debt on lane 0 and Rd's WAR debt on lane 2 — beside the source's
    // producer W on lane 1. Survivors ascend by lane: T0, W, Rd.
    plan.queue.push_back(
        make_copy(0, 0, ChunkLoc{1, 0}, ChunkLoc{2, 0}));
    PlanDerivation const r = d.run(plan);

    OpRef const T0 = make_op_ref(0, 0);
    OpRef const Rd = make_op_ref(0, 1);
    OpRef const W = make_op_ref(0, 2);
    OpRef const CP = make_op_ref(0, 3);

    // The copy owes three cross-lane debts: both destination users and
    // the source producer — WAR and WAW protection for the destination
    // included, per lane.
    CHECK((r.owed_set[3] == std::vector<OpRef>{T0, W, Rd}));
    CHECK(r.owed_set[3].size() == 3);

    // The touchers and the producer all mint (each first owed
    // cross-lane here or before); the copy itself is owed by nobody.
    CHECK(r.mints[0] && r.mints[1] && r.mints[2]);
    CHECK(!r.mints[3]);

    // And the copy's own effect: it becomes (2,0)'s producer and its
    // latest user on chan(1,2), leaving the earlier user facts intact.
    CHECK(d.state().producer(ChunkLoc{2, 0}) == CP);
    CHECK(d.state().user(ChunkLoc{2, 0}, lane_of_op(plan.queue[3], inv)) ==
        CP);
    CHECK(d.state().user(ChunkLoc{2, 0}, 0) == T0);
    CHECK(d.state().user(ChunkLoc{2, 0}, 2) == Rd);
    CHECK(r.late_records.empty());
}

// ---- the domination fold  and the per-consumer mint ----

void test_domination_fold()
{
    NodeInventory const inv = make_inventory(true);
    Derivation d(inv);

    Plan plan;
    plan.plan_seq = 0;
    // Three writers on ONE lane (exec(1)), one chunk each, issued in
    // the order Wa, Wc, Wb — Wb is the LATEST of the three.
    plan.queue.push_back(
        make_tile_op(1, 1, {opnd(0, 0, Access::Write, 1, 0)})); // Wa
    plan.queue.push_back(
        make_tile_op(1, 1, {opnd(0, 1, Access::Write, 1, 1)})); // Wc
    plan.queue.push_back(
        make_tile_op(1, 1, {opnd(0, 2, Access::Write, 1, 2)})); // Wb
    // Reader A on exec(0) reads all three chunks: three debts, all on
    // exec(1) — operands ordered so the adds are NOT in issue order
    // (Wa, Wc, Wb), pinning the fold's position comparison.
    plan.queue.push_back(make_tile_op(2, 0,
        {opnd(0, 0, Access::Read, 1, 0), opnd(0, 1, Access::Read, 1, 1),
            opnd(0, 2, Access::Read, 1, 2)}));
    // Reader B on exec(2) (host) reads only Wa's chunk.
    plan.queue.push_back(
        make_tile_op(2, 2, {opnd(0, 0, Access::Read, 1, 0)}));
    // Reader C on exec(2) reads Wb's chunk first, then Wa's: the newer
    // ref added first must survive the older add that follows.
    plan.queue.push_back(make_tile_op(2, 2,
        {opnd(0, 2, Access::Read, 1, 2), opnd(0, 0, Access::Read, 1, 0)}));
    PlanDerivation const r = d.run(plan);

    OpRef const Wa = make_op_ref(0, 0); // p0, kept by reader B's fold
    OpRef const Wb = make_op_ref(0, 2); // p2, the latest — the survivor

    // ONE survivor per other lane: the latest, Wb. Dominated edges are
    // never named — Wa and Wc appear in no folded set of A or C.
    CHECK(r.owed_set[3] == std::vector<OpRef>{Wb});
    CHECK(r.owed_set[3].size() == 1);
    CHECK(r.owed_set[5] == std::vector<OpRef>{Wb});
    CHECK(r.owed_set[5].size() == 1);

    // Domination is PER-CONSUMER: B's fold keeps Wa alone (its only
    // debt), so Wa's event mints for B though A's fold discarded it.
    // B and C are consecutive on exec(2) — one segment — and C's Wb is
    // a NEW max on exec(1) (Wb > Wa): BOTH rows keep their
    // own refs, each wait at its own member (the intra-segment drop
    // only removes refs dominated by an earlier member's kept ref).
    CHECK(r.owed_set[4] == std::vector<OpRef>{Wa});

    // The mint marks follow the folds: Wb minted (A's survivor), Wa
    // minted (B's survivor), Wc minted by NOBODY — a dominated
    // producer's event is never minted for the consumer whose fold
    // discarded it, and no other consumer kept it.
    CHECK(r.mints[2]); // Wb
    CHECK(r.mints[0]); // Wa — kept by another consumer
    CHECK(!r.mints[1]); // Wc — dominated for every consumer
    CHECK(r.late_records.empty());
}

// ---- the mint iff  ----

void test_mint_iff()
{
    NodeInventory const inv = make_inventory(true);
    Derivation d(inv);

    Plan plan;
    plan.plan_seq = 0;
    plan.queue.push_back(
        make_tile_op(1, 1, {opnd(0, 0, Access::Write, 1, 0)})); // P
    plan.queue.push_back(
        make_tile_op(1, 1, {opnd(0, 1, Access::Write, 1, 1)})); // Q
    plan.queue.push_back(
        make_tile_op(2, 0, {opnd(0, 0, Access::Read, 1, 0)})); // Ca
    plan.queue.push_back(
        make_tile_op(2, 2, {opnd(0, 0, Access::Read, 1, 0)})); // Cb
    PlanDerivation const r = d.run(plan);

    OpRef const P = make_op_ref(0, 0); // p0, owed cross-lane
    // p1 is Q, the never-owed producer: minted by nobody (mints[1]).

    // Both consumers owe exactly P, cross-lane, post-fold. The
    // consumers are singleton segments — the intra-segment drop never
    // fires, each row keeps its own ref at its own position.
    CHECK(r.owed_set[2] == std::vector<OpRef>{P});
    CHECK(r.owed_set[3] == std::vector<OpRef>{P});

    // A never-owed producer mints NOTHING; a cross-lane owed producer
    // mints — and the consumers themselves are owed by nobody.
    CHECK(r.mints[0]);
    CHECK(!r.mints[1]);
    CHECK(!r.mints[2]);
    CHECK(!r.mints[3]);

    // Exactly once: the executor mints per the marking, and two
    // consumers waiting one event mint it still once — never two on
    // one op.
    EventRegistry reg;
    std::uint32_t next_handle = 1;
    for (std::size_t p = 0; p < plan.queue.size(); ++p)
    {
        if (r.mints[p])
        {
            reg.insert(make_op_ref(0, p),
                EventRegistry::Entry{next_handle++,
                    lane_of_op(plan.queue[p], inv)});
        }
    }
    CHECK(reg.map.size() == 1); // P only — Q minted nothing
    EventRegistry::Entry const &e = reg.resolve(P);
    CHECK(e.handle == 1);
    CHECK(e.owner == inv.lane_of_exec(1));
}

// ---- late records across a plan boundary  ----

void test_late_records()
{
    NodeInventory const inv = make_inventory(true);
    Derivation d(inv);

    // Plan 1: the producer, owed by nobody yet — no mint at its own
    // pass, never speculative.
    Plan p1;
    p1.plan_seq = 1;
    p1.queue.push_back(
        make_tile_op(1, 1, {opnd(0, 0, Access::Write, 1, 0)}));
    PlanDerivation const r1 = d.run(p1);
    OpRef const P = make_op_ref(1, 0);
    CHECK(!r1.mints[0]);
    CHECK(r1.late_records.empty());

    // Plan 2: the consumer owes P — a cross-plan debt over the same
    // run-scoped facts. The mint is marked at THIS pass as a LATE
    // RECORD into P's own lane; plan 2's own op mints nothing.
    Plan p2;
    p2.plan_seq = 2;
    p2.queue.push_back(
        make_tile_op(2, 0, {opnd(0, 0, Access::Read, 1, 0)}));
    PlanDerivation const r2 = d.run(p2);
    CHECK(r2.owed_set[0] == std::vector<OpRef>{P});
    CHECK(r2.late_records.size() == 1);
    CHECK(r2.late_records[0].op == P);
    CHECK(r2.late_records[0].owner == inv.lane_of_exec(1));
    CHECK(!r2.mints[0]);

    // Plan 3: a second consumer owes P again — the mint already
    // exists, several consumers wait one event, and never two mint on
    // one op: no second late record.
    Plan p3;
    p3.plan_seq = 3;
    p3.queue.push_back(
        make_tile_op(2, 2, {opnd(0, 0, Access::Read, 1, 0)}));
    PlanDerivation const r3 = d.run(p3);
    CHECK(r3.owed_set[0] == std::vector<OpRef>{P});
    CHECK(r3.late_records.empty());

    // The carried facts: plan 2's and plan 3's readers are the chunk's
    // users on their lanes; P is still its producer (no later writer).
    CHECK(d.state().producer(ChunkLoc{1, 0}) == P);
    CHECK(d.state().user(ChunkLoc{1, 0}, 0) == make_op_ref(2, 0));
    CHECK(d.state().user(ChunkLoc{1, 0}, 2) == make_op_ref(3, 0));

    // A plan out of run-queue order is loud .
    Plan stale;
    stale.plan_seq = 2;
    CHECK(throws_loud([&] { d.run(stale); }, "run-queue order"));
}

// ---- registry-miss on a hand-broken wait  ----

void test_registry_miss()
{
    EventRegistry reg;

    // The hand-broken wait: the consumer's folded set names an op the
    // registry carries no entry for — the engine-invariant abort, loud
    // with the fault's name and the ref rendered.
    OpRef const broken = make_op_ref(4, 2);
    CHECK(throws_loud([&] { (void)reg.resolve(broken); }, "registry-miss"));
    CHECK(throws_loud([&] { (void)reg.resolve(broken); }, "position=2"));

    // A present entry resolves to its mint and its minting lane.
    OpRef const present = make_op_ref(4, 0);
    reg.insert(present, EventRegistry::Entry{7, 1});
    EventRegistry::Entry const &e = reg.resolve(present);
    CHECK(e.handle == 7);
    CHECK(e.owner == 1);
    CHECK(throws_loud([&] { (void)reg.resolve(broken); }, "registry-miss"));

    // Never two events on one op: the duplicate mint is a registry
    // defect, loud.
    CHECK(throws_loud(
        [&] { reg.insert(present, EventRegistry::Entry{8, 1}); }));
}

// ---- sweep displacement  ----

void test_sweep_displacement()
{
    NodeInventory const inv = make_inventory(true);
    Derivation d(inv);
    EventRegistry reg;

    // Plan 1: W0 writes (1,0); R reads it (minting W0); the slot is
    // UNLINKed and LINKed to a new tile (the epoch boundary); W1
    // rewrites it — consulting R's USER fact, so R mints too.
    Plan p1;
    p1.plan_seq = 1;
    p1.queue.push_back(
        make_tile_op(1, 1, {opnd(0, 0, Access::Write, 1, 0)})); // W0
    p1.queue.push_back(
        make_tile_op(2, 0, {opnd(0, 0, Access::Read, 1, 0)})); // R
    p1.queue.push_back(make_unlink(0, 0, 1, 0));
    p1.queue.push_back(make_link(9, 0, 1, 0));
    p1.queue.push_back(
        make_tile_op(1, 1, {opnd(9, 0, Access::Write, 1, 0)})); // W1
    PlanDerivation const r1 = d.run(p1);
    OpRef const W0 = make_op_ref(1, 0);
    OpRef const R = make_op_ref(1, 1);
    OpRef const W1 = make_op_ref(1, 4);
    CHECK(r1.owed_set[1] == std::vector<OpRef>{W0}); // R owes W0
    // W1 owes R: the USER fact that survived the whole epoch.
    CHECK(r1.owed_set[4] == std::vector<OpRef>{R});
    CHECK(r1.mints[0] && r1.mints[1] && !r1.mints[4]);

    // The executor mints per the marking, then sweeps at queue close.
    reg.insert(W0, EventRegistry::Entry{1, inv.lane_of_exec(1)});
    reg.insert(R, EventRegistry::Entry{2, inv.lane_of_exec(0)});
    std::vector<std::pair<OpRef, EventRegistry::Entry>> displaced;
    reg.sweep(d.state(), displaced);
    // W0 is named by NO fact anymore (W1 displaced its producer and
    // user facts): displaced, dropped. R is still lane 0's USER of the
    // chunk — a fact from the OLD epoch, pre-unlink — so it survives.
    CHECK(reg.map.size() == 1);
    CHECK(displaced.size() == 1);
    CHECK(displaced[0].first == W0);
    CHECK(displaced[0].second.handle == 1);
    CHECK(reg.resolve(R).handle == 2);
    CHECK(throws_loud([&] { (void)reg.resolve(W0); }, "registry-miss"));

    // Plan 2: an unrelated write — R's fact stands untouched, so R
    // survives the NEXT sweep too, even after its plan closed.
    Plan p2;
    p2.plan_seq = 2;
    p2.queue.push_back(
        make_tile_op(1, 1, {opnd(0, 1, Access::Write, 1, 1)}));
    PlanDerivation const r2 = d.run(p2);
    CHECK(r2.owed_set[0].empty());
    displaced.clear();
    reg.sweep(d.state(), displaced);
    CHECK(reg.map.size() == 1);
    CHECK(displaced.empty());
    CHECK(reg.resolve(R).handle == 2);

    // Plan 3: R2 reads the chunk on lane 0 — R2's own USER fact
    // displaces R's, and only now does no fact name R: the sweep drops
    // it. W1 (owed by R2 cross-lane) mints and stays — producer and
    // lane-1 user both name it.
    Plan p3;
    p3.plan_seq = 3;
    p3.queue.push_back(
        make_tile_op(2, 0, {opnd(9, 0, Access::Read, 1, 0)})); // R2
    PlanDerivation const r3 = d.run(p3);
    OpRef const R2 = make_op_ref(3, 0);
    CHECK(r3.owed_set[0] == std::vector<OpRef>{W1});
    CHECK(r3.mints[0] == false); // R2 is owed by nobody
    reg.insert(W1, EventRegistry::Entry{3, inv.lane_of_exec(1)});
    displaced.clear();
    reg.sweep(d.state(), displaced);
    CHECK(reg.map.size() == 1);
    CHECK(displaced.size() == 1);
    CHECK(displaced[0].first == R);
    CHECK(displaced[0].second.handle == 2);
    CHECK(reg.resolve(W1).handle == 3);
    CHECK(throws_loud([&] { (void)reg.resolve(R); }, "registry-miss"));
    CHECK(throws_loud([&] { (void)reg.resolve(R2); }, "registry-miss"));
}

} // namespace

int main()
{
    test_update_tables();
    test_user_never_resets();
    test_same_lane_elision();
    test_copy_dst_users();
    test_domination_fold();
    test_mint_iff();
    test_late_records();
    test_registry_miss();
    test_sweep_displacement();

    if (g_checks_failed != 0)
    {
        std::cerr << "dependency: " << g_checks_failed
                  << " check(s) failed\n";
        return 1;
    }
    std::cout << "dependency: all checks passed\n";
    return 0;
}
