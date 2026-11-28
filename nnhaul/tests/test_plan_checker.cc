/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file tests the
 * stateless plan checker: the fold fault table, one hand-built plan per
 * row.
 *
 * @file tests/test_plan_checker.cc
 * The checker's suite : fold fault-table coverage —
 * every fault name gets a hand-built plan triggering exactly that row
 * and no earlier row: every per-op row of the four dispatch tables
 * (malformed-plan variants, unknown-id,
 * unknown-tile, index-out-of-range, double-link, double-binding,
 * unlink-free, identity-mismatch, copy-identity-mismatch both
 * endpoints, copy-same-node, copy-no-channel, copy-from-outdated,
 * copy-to-actual, task-not-runnable, operand-node-mismatch,
 * access-unlinked, read-outdated, duplicate-operand, duplicate-write),
 * the entry-state rows (unknown-id, unknown-tile,
 * index-out-of-range, double-binding, double-link), the
 * correspondence faults (task-invented, task-missing,
 * task-order-violation, operand-coverage-violation both the gap and
 * the outside entry, access-mode-mismatch), and boundary-discontinuity
 * via a supplied chain — each asserting the census names the fault and
 * the position. The repeated-TaskId regression (a production
 * trace's census): one TaskId at two sequence positions with
 * overlapping tiles and flipped accesses PASSES — the checks are per
 * OCCURRENCE, each collapsed run against its own sequence entry —
 * while the true per-occurrence mismatch still fires, and a repeat's
 * missing second occurrence is a strict prefix. Positive cases:
 * a complete legal plan passing with the
 * exact hand-computed exit state (flags after writes, a copy, replica
 * outdating, an unlink), a chained plan passing against its
 * predecessor's exit, the plan leaving a passed check byte for byte
 * unchanged, and the loud caller-defect surface (null mandatory
 * inputs, a task sequence off the tile table). Beside the checker, the
 * tile table's own registration facts (count and payload bytes of one
 * registration) and the chunk span they feed (the occupied span of every
 * chunk index is min(class, nbytes -
 * index*class) — full chunk the class, tail chunk the remainder).
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "../src/plan_checker.hh"

// check.hh's shared macro pulls the OLD public header (nnhaul/nnhaul.hh)
// beside the new contract types, and the two disagree on nnhaul::Access
// today — this suite needs no runtime guards, so the CHECK macro stands
// here alone, same shape as check.hh's (as test_binding_table.cc).
#include "nnhaul/defs.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
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

// A loud-throw probe: runs f, true iff it threw std::runtime_error
// carrying needle (the defect must be greppable).
template <typename F>
bool throws_loud(F &&f, char const *needle)
{
    try
    {
        f();
    }
    catch (std::runtime_error const &e)
    {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    catch (...)
    {
        return false;
    }
    return false;
}

// ---- the standard world: 2 devices + host, no peer channels ----

// Nodes: 0, 1 devices, 2 host . Cards 8 chunks per node; peer
// matrix zero — the channels are exactly the four device<->host pairs
// (0->2 1->2 2->0 2->1), so copy-no-channel has a device pair to name.
// Tiles: A, B, C of one chunk; W of two. Tasks: T1 (device 0), T2
// (devices 0 and 1), TH (the host), TX runnable on device 0 but never
// of the default task sequences (the task-invented row's task).
NodeId const kDev0 = 0;
NodeId const kDev1 = 1;
NodeId const kHost = 2;
TileId const kA = 0;
TileId const kB = 1;
TileId const kC = 2;
TileId const kW = 3;
TaskId const kT1 = 0;
TaskId const kT2 = 1;
TaskId const kTH = 2;
TaskId const kTX = 9;

struct World
{
    NodeInventory inv;
    TileTable tiles;
    ExecutabilityTable runnable;

    World()
        : inv(2, {8, 8, 8}, {0, 0, 0, 0})
    {
        tiles.add(kA, chunk_class_bytes());            // n = 1
        tiles.add(kB, chunk_class_bytes());            // n = 1
        tiles.add(kC, chunk_class_bytes());            // n = 1
        tiles.add(kW, 2 * chunk_class_bytes());        // n = 2
        runnable.add(kT1, {kDev0});
        runnable.add(kT2, {kDev0, kDev1});
        runnable.add(kTH, {kHost});
        runnable.add(kTX, {kDev0}); // runnable, never of a sequence
    }
};

// ---- the kit: bindings, ops, sequences, the check itself ----

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

Operand opnd(TileId tile, ChunkIndex index, Access access, NodeId node,
    ChunkId chunk)
{
    Operand e;
    e.tile = tile;
    e.index = index;
    e.access = access;
    e.node = node;
    e.chunk = chunk;
    return e;
}

SeqTask seq_task(TaskId task, std::vector<TileAccess> operands)
{
    SeqTask t;
    t.task = task;
    t.operands = std::move(operands);
    return t;
}

TileAccess ta(TileId tile, Access access)
{
    TileAccess a;
    a.tile = tile;
    a.access = access;
    return a;
}

CheckResult run_check(World const &w, Plan const &plan,
    std::vector<SeqTask> const &sequence,
    std::optional<std::vector<EntryBinding>> chain = std::nullopt)
{
    CheckInput in;
    in.plan = &plan;
    in.inventory = &w.inv;
    in.tiles = &w.tiles;
    in.runnable = &w.runnable;
    in.task_sequence = &sequence;
    in.chain = std::move(chain);
    return check(in);
}

// The fault assertion: the verdict names the fault, sits at the site,
// carries the offending op verbatim when one exists, and the census
// names the fault AND the position . For
// a Queue finding the census's ops prefix is everything folded before
// the offending op, entry for entry.
void expect_fault(CheckResult const &r, std::string const &name,
    CheckSite site, std::size_t position, PlanOp const *op = nullptr)
{
    CHECK(!r.pass);
    CHECK(r.fault == name);
    CHECK(r.site == site);
    CHECK(r.position == position);
    CHECK(r.exit.empty());
    CHECK(r.census.has_value());
    if (!r.census.has_value())
    {
        return;
    }
    Census const &c = *r.census;
    CHECK(c.fault.find(name) != std::string::npos);
    std::string where;
    if (site == CheckSite::Queue)
    {
        where = "at op " + std::to_string(position);
    }
    else if (site == CheckSite::Entry)
    {
        where = "at entry";
    }
    else
    {
        where = "at close";
    }
    CHECK(c.fault.find(where) != std::string::npos);
    CHECK(!c.fold.empty());        // the fold render, every node
    CHECK(c.applied.empty());      // execution time only
    CHECK(c.issue_prefix.empty()); // execution time only
    if (site == CheckSite::Queue)
    {
        CHECK(c.prefix.size() == position);
    }
    else if (site == CheckSite::Entry)
    {
        CHECK(c.prefix.empty());
    }
    if (op != nullptr)
    {
        CHECK(r.op.has_value());
        if (r.op.has_value())
        {
            CHECK(*r.op == *op);
        }
        CHECK(c.op.has_value());
        if (c.op.has_value())
        {
            CHECK(*c.op == *op);
        }
    }
}

// ---- positive: a complete legal plan, exact hand-computed exit ----

void positive_full_plan()
{
    World w;
    Plan p;
    p.plan_seq = 7;
    // Cold start: no entry, no chain.
    p.queue.push_back(make_link(kA, 0, kDev0, 0)); // (A,0)@dev0 outdated
    p.queue.push_back(make_link(kA, 0, kHost, 0)); // replica on host
    // T1 writes (A,0) on dev0: the binding turns ACTUAL, the host
    // replica is OUTDATED (already is).
    p.queue.push_back(make_tile_op(kT1, kDev0,
        {opnd(kA, 0, Access::Write, kDev0, 0)}));
    // Host-staged replication: dev0 -> host over the always-channel;
    // dst was outdated, turns ACTUAL; both replicas now actual.
    p.queue.push_back(
        make_copy(kA, 0, ChunkLoc{kDev0, 0}, ChunkLoc{kHost, 0}));
    // The host task reads and rewrites (A,0) on the host replica: legal
    // against the pre-op fold, then the dev0 replica turns OUTDATED.
    p.queue.push_back(make_tile_op(kTH, kHost,
        {opnd(kA, 0, Access::ReadWrite, kHost, 0)}));
    // The dev0 binding drops; UNLINK is indifferent to the flag.
    p.queue.push_back(make_unlink(kA, 0, kDev0, 0));

    std::vector<SeqTask> sequence{
        seq_task(kT1, {ta(kA, Access::Write)}),
        seq_task(kTH, {ta(kA, Access::ReadWrite)})};

    Plan const before = p; // byte-for-byte witness
    CheckResult r = run_check(w, p, sequence);
    CHECK(r.pass);
    CHECK(r.fault.empty());
    CHECK(!r.census.has_value());
    // The exit: only the host binding, ACTUAL — dev0's chunk freed by
    // the unlink, the write's replica outdating visible in history.
    std::vector<EntryBinding> expected{eb(kHost, 0, kA, 0, true)};
    CHECK(r.exit.size() == expected.size());
    if (r.exit.size() == expected.size())
    {
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            CHECK(r.exit[i] == expected[i]);
        }
    }
    // The plan that leaves a passed check is the plan that entered it.
    CHECK(p.plan_seq == before.plan_seq);
    CHECK(p.entry.size() == before.entry.size());
    CHECK(p.queue.size() == before.queue.size());
    for (std::size_t i = 0; i < p.queue.size(); ++i)
    {
        CHECK(p.queue[i] == before.queue[i]);
    }
}

// ---- positive: the chained boundary, exact and passing ----

void positive_chained()
{
    World w;
    // The predecessor's exit (positive_full_plan's hand computation).
    std::vector<EntryBinding> pred_exit{eb(kHost, 0, kA, 0, true)};

    Plan p;
    p.plan_seq = 8;
    p.entry = pred_exit;
    p.queue.push_back(make_link(kB, 0, kDev0, 0)); // freed last plan
    // T1 writes B on dev0: actual, no other replica to outdate.
    p.queue.push_back(make_tile_op(kT1, kDev0,
        {opnd(kB, 0, Access::Write, kDev0, 0)}));

    std::vector<SeqTask> sequence{seq_task(kT1, {ta(kB, Access::Write)})};

    // The chaining check passes: the entry fold equals the supplied
    // chain over ALL nodes; the plan folds to a two-binding exit.
    CheckResult r = run_check(w, p, sequence, pred_exit);
    CHECK(r.pass);
    std::vector<EntryBinding> expected{eb(kDev0, 0, kB, 0, true),
        eb(kHost, 0, kA, 0, true)};
    CHECK(r.exit.size() == expected.size());
    if (r.exit.size() == expected.size())
    {
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            CHECK(r.exit[i] == expected[i]);
        }
    }

    // The same plan against a chain one flag off: boundary-discontinuity
    // at entry, before any op.
    std::vector<EntryBinding> flipped{eb(kHost, 0, kA, 0, false)};
    r = run_check(w, p, sequence, flipped);
    expect_fault(r, "boundary-discontinuity", CheckSite::Entry, 0);
    CHECK(r.census->prefix.empty());
    // The census echoes both sides: the entry state and the chain.
    CHECK(r.census->entry == p.entry);
    CHECK(r.census->inputs.find("chain=[") != std::string::npos);

    // A chain naming one binding more is a discontinuity too.
    std::vector<EntryBinding> extra(pred_exit);
    extra.push_back(eb(kDev0, 1, kB, 0, false));
    r = run_check(w, p, sequence, extra);
    expect_fault(r, "boundary-discontinuity", CheckSite::Entry, 0);
}

// ---- malformed-plan: the shape rows of all four tables ----

void malformed_plan_variants()
{
    World w;
    std::vector<SeqTask> sequence{seq_task(kT1, {ta(kA, Access::Write)})};

    { // LINK carrying operands — not exactly the four fields.
        Plan p;
        PlanOp op = make_link(kA, 0, kDev0, 0);
        op.operands.push_back(opnd(kA, 0, Access::Read, kDev0, 0));
        p.queue.push_back(op);
        expect_fault(run_check(w, p, sequence), "malformed-plan",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // UNLINK carrying operands.
        Plan p;
        PlanOp op = make_unlink(kA, 0, kDev0, 0);
        op.operands.push_back(opnd(kA, 0, Access::Read, kDev0, 0));
        p.queue.push_back(op);
        expect_fault(run_check(w, p, sequence), "malformed-plan",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // COPY carrying operands.
        Plan p;
        PlanOp op = make_copy(
            kA, 0, ChunkLoc{kDev0, 0}, ChunkLoc{kHost, 0});
        op.operands.push_back(opnd(kA, 0, Access::Read, kDev0, 0));
        p.queue.push_back(op);
        expect_fault(run_check(w, p, sequence), "malformed-plan",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // TILE_OP with an access outside the three modes.
        Plan p;
        PlanOp op = make_tile_op(kT1, kDev0,
            {opnd(kA, 0, static_cast<Access>(7), kDev0, 0)});
        p.queue.push_back(op);
        expect_fault(run_check(w, p, sequence), "malformed-plan",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // A queue entry of no kind at all: the queue contains EXACTLY
      // the four kinds .
        Plan p;
        PlanOp op;
        op.kind = static_cast<PlanOpKind>(200);
        p.queue.push_back(op);
        expect_fault(run_check(w, p, sequence), "malformed-plan",
            CheckSite::Queue, 0, &p.queue[0]);
    }
}

// ---- unknown-id: every ids row of the four tables and the entry ----

void unknown_id_rows()
{
    World w;
    std::vector<SeqTask> sequence{seq_task(kT1, {ta(kA, Access::Write)})};

    { // LINK: node off the inventory.
        Plan p;
        p.queue.push_back(make_link(kA, 0, 7, 0));
        expect_fault(run_check(w, p, sequence), "unknown-id",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // LINK: chunk off the node's pool.
        Plan p;
        p.queue.push_back(make_link(kA, 0, kDev0, 8));
        expect_fault(run_check(w, p, sequence), "unknown-id",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // UNLINK: node off the inventory.
        Plan p;
        p.queue.push_back(make_unlink(kA, 0, 7, 0));
        expect_fault(run_check(w, p, sequence), "unknown-id",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // COPY: the source's node off the inventory.
        Plan p;
        p.queue.push_back(
            make_copy(kA, 0, ChunkLoc{7, 0}, ChunkLoc{kHost, 0}));
        expect_fault(run_check(w, p, sequence), "unknown-id",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // COPY: the source's chunk off the pool.
        Plan p;
        p.queue.push_back(
            make_copy(kA, 0, ChunkLoc{kDev0, 8}, ChunkLoc{kHost, 0}));
        expect_fault(run_check(w, p, sequence), "unknown-id",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // COPY: the destination's chunk off the pool.
        Plan p;
        p.queue.push_back(
            make_copy(kA, 0, ChunkLoc{kDev0, 0}, ChunkLoc{kHost, 8}));
        expect_fault(run_check(w, p, sequence), "unknown-id",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // TILE_OP: the op's node off the inventory.
        Plan p;
        p.queue.push_back(make_tile_op(kT1, 7, {}));
        expect_fault(run_check(w, p, sequence), "unknown-id",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // TILE_OP: an operand's node off the inventory.
        Plan p;
        p.queue.push_back(
            make_tile_op(kT1, kDev0, {opnd(kA, 0, Access::Read, 9, 0)}));
        expect_fault(run_check(w, p, sequence), "unknown-id",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // TILE_OP: an operand's chunk off the pool.
        Plan p;
        p.queue.push_back(make_tile_op(
            kT1, kDev0, {opnd(kA, 0, Access::Read, kDev0, 9)}));
        expect_fault(run_check(w, p, sequence), "unknown-id",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // Entry: the binding's node off the inventory.
        Plan p;
        p.entry.push_back(eb(7, 0, kA, 0, false));
        expect_fault(
            run_check(w, p, sequence), "unknown-id", CheckSite::Entry, 0);
    }
    { // Entry: the binding's chunk off the pool — the SECOND binding,
      // after a valid one folded in.
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, true));
        p.entry.push_back(eb(kDev0, 8, kB, 0, false));
        expect_fault(
            run_check(w, p, sequence), "unknown-id", CheckSite::Entry, 1);
    }
}

// ---- unknown-tile: the tile rows of the four tables and the entry ----

void unknown_tile_rows()
{
    World w;
    std::vector<SeqTask> sequence{seq_task(kT1, {ta(kA, Access::Write)})};
    TileId const kUnknown = 99;

    { // LINK
        Plan p;
        p.queue.push_back(make_link(kUnknown, 0, kDev0, 0));
        expect_fault(run_check(w, p, sequence), "unknown-tile",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // UNLINK
        Plan p;
        p.queue.push_back(make_unlink(kUnknown, 0, kDev0, 0));
        expect_fault(run_check(w, p, sequence), "unknown-tile",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // COPY (both endpoints of the pools; the tile is the row).
        Plan p;
        p.queue.push_back(make_copy(
            kUnknown, 0, ChunkLoc{kDev0, 0}, ChunkLoc{kHost, 0}));
        expect_fault(run_check(w, p, sequence), "unknown-tile",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // TILE_OP operand
        Plan p;
        p.queue.push_back(make_tile_op(
            kT1, kDev0, {opnd(kUnknown, 0, Access::Read, kDev0, 0)}));
        expect_fault(run_check(w, p, sequence), "unknown-tile",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // Entry
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kUnknown, 0, false));
        expect_fault(
            run_check(w, p, sequence), "unknown-tile", CheckSite::Entry, 0);
    }
}

// ---- index-out-of-range: the index rows, W being a 2-chunk tile ----

void index_out_of_range_rows()
{
    World w;
    std::vector<SeqTask> sequence{seq_task(kT1, {ta(kA, Access::Write)})};

    { // LINK
        Plan p;
        p.queue.push_back(make_link(kW, 2, kDev0, 0));
        expect_fault(run_check(w, p, sequence), "index-out-of-range",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // UNLINK
        Plan p;
        p.queue.push_back(make_unlink(kW, 2, kDev0, 0));
        expect_fault(run_check(w, p, sequence), "index-out-of-range",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // COPY
        Plan p;
        p.queue.push_back(make_copy(
            kW, 2, ChunkLoc{kDev0, 0}, ChunkLoc{kHost, 0}));
        expect_fault(run_check(w, p, sequence), "index-out-of-range",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // TILE_OP operand
        Plan p;
        p.queue.push_back(make_tile_op(
            kT1, kDev0, {opnd(kW, 2, Access::Read, kDev0, 0)}));
        expect_fault(run_check(w, p, sequence), "index-out-of-range",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // Entry
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kW, 2, false));
        expect_fault(run_check(w, p, sequence), "index-out-of-range",
            CheckSite::Entry, 0);
    }
}

// ---- the linkage rows: link, unlink, and the entry's own ----

void linkage_rows()
{
    World w;
    std::vector<SeqTask> sequence{seq_task(kT1, {ta(kA, Access::Write)})};

    { // double-link: LINK onto a chunk already linked (row 5 before
      // the one-per-node row, whatever the logical chunks).
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, true));
        p.queue.push_back(make_link(kA, 0, kDev0, 0));
        expect_fault(run_check(w, p, sequence), "double-link",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // double-binding: the logical chunk already bound on the node,
      // the named chunk free.
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, true));
        p.queue.push_back(make_link(kA, 0, kDev0, 1));
        expect_fault(run_check(w, p, sequence), "double-binding",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // double-binding through the queue's own LINKs: the finding is
      // the second op's, position 1.
        Plan p;
        p.queue.push_back(make_link(kA, 0, kDev0, 0));
        p.queue.push_back(make_link(kA, 0, kDev0, 1));
        expect_fault(run_check(w, p, sequence), "double-binding",
            CheckSite::Queue, 1, &p.queue[1]);
    }
    { // unlink-free: nothing linked there.
        Plan p;
        p.queue.push_back(make_unlink(kA, 0, kDev0, 0));
        expect_fault(run_check(w, p, sequence), "unlink-free",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // identity-mismatch: the chunk bound to another tile.
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, true));
        p.queue.push_back(make_unlink(kB, 0, kDev0, 0));
        expect_fault(run_check(w, p, sequence), "identity-mismatch",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // identity-mismatch: a TILE_OP names a chunk that holds another
      // logical chunk (bytes of the wrong tile must never be executed).
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, true));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kB, 0, Access::Read, kDev0, 0)}));
        std::vector<SeqTask> const sequence{
            seq_task(kT1, {ta(kB, Access::Read)})};
        expect_fault(run_check(w, p, sequence), "identity-mismatch",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // identity-mismatch: the index not the binding's own.
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kW, 1, true));
        p.queue.push_back(make_unlink(kW, 0, kDev0, 0));
        expect_fault(run_check(w, p, sequence), "identity-mismatch",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // Entry double-binding: one logical chunk, two chunks of one
      // node — the second binding is the finding.
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, false));
        p.entry.push_back(eb(kDev0, 1, kA, 0, false));
        expect_fault(run_check(w, p, sequence), "double-binding",
            CheckSite::Entry, 1);
    }
    { // Entry double-link: two bindings naming one chunk with
      // different logical chunks.
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, false));
        p.entry.push_back(eb(kDev0, 0, kB, 0, false));
        expect_fault(run_check(w, p, sequence), "double-link",
            CheckSite::Entry, 1);
    }
}

// ---- the COPY table's own rows: identity, endpoints, channel, flags ----

void copy_rows()
{
    World w;
    std::vector<SeqTask> sequence{seq_task(kT1, {ta(kA, Access::Write)})};

    { // copy-identity-mismatch: the source holds another logical
      // chunk.
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, true));
        p.entry.push_back(eb(kHost, 0, kA, 0, false));
        p.queue.push_back(
            make_copy(kB, 0, ChunkLoc{kDev0, 0}, ChunkLoc{kHost, 0}));
        expect_fault(run_check(w, p, sequence),
            "copy-identity-mismatch", CheckSite::Queue, 0, &p.queue[0]);
    }
    { // copy-identity-mismatch: the destination holds another logical
      // chunk (the source's identity is fine).
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, true));
        p.entry.push_back(eb(kHost, 0, kB, 0, true));
        p.queue.push_back(
            make_copy(kA, 0, ChunkLoc{kDev0, 0}, ChunkLoc{kHost, 0}));
        expect_fault(run_check(w, p, sequence),
            "copy-identity-mismatch", CheckSite::Queue, 0, &p.queue[0]);
    }
    { // copy-identity-mismatch: a free destination is a mismatch like
      // any other — "linked exactly to".
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, true));
        p.queue.push_back(
            make_copy(kA, 0, ChunkLoc{kDev0, 0}, ChunkLoc{kHost, 0}));
        expect_fault(run_check(w, p, sequence),
            "copy-identity-mismatch", CheckSite::Queue, 0, &p.queue[0]);
    }
    { // copy-same-node: both endpoints one node — reachable as one
      // and the same binding, since one logical chunk holds at most
      // one binding per node .
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, true));
        p.queue.push_back(make_copy(
            kA, 0, ChunkLoc{kDev0, 0}, ChunkLoc{kDev0, 0}));
        expect_fault(run_check(w, p, sequence), "copy-same-node",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // copy-no-channel: dev0 -> dev1 against the zero peer matrix.
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, true));
        p.entry.push_back(eb(kDev1, 0, kA, 0, false));
        p.queue.push_back(
            make_copy(kA, 0, ChunkLoc{kDev0, 0}, ChunkLoc{kDev1, 0}));
        expect_fault(run_check(w, p, sequence), "copy-no-channel",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // copy-from-outdated: the source's flag is the row.
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, false));
        p.entry.push_back(eb(kHost, 0, kA, 0, false));
        p.queue.push_back(
            make_copy(kA, 0, ChunkLoc{kDev0, 0}, ChunkLoc{kHost, 0}));
        expect_fault(run_check(w, p, sequence), "copy-from-outdated",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // copy-to-actual: the destination already holds the bytes.
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, true));
        p.entry.push_back(eb(kHost, 0, kA, 0, true));
        p.queue.push_back(
            make_copy(kA, 0, ChunkLoc{kDev0, 0}, ChunkLoc{kHost, 0}));
        expect_fault(run_check(w, p, sequence), "copy-to-actual",
            CheckSite::Queue, 0, &p.queue[0]);
    }
}

// ---- the TILE_OP table's own rows ----

void tileop_rows()
{
    World w;
    std::vector<SeqTask> sequence{seq_task(kT1, {ta(kA, Access::Write)})};

    { // task-not-runnable: T1 runs on dev0 only, the op names the host.
        Plan p;
        p.entry.push_back(eb(kHost, 0, kA, 0, true));
        p.queue.push_back(make_tile_op(
            kT1, kHost, {opnd(kA, 0, Access::Read, kHost, 0)}));
        expect_fault(run_check(w, p, sequence), "task-not-runnable",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // operand-node-mismatch: the operand lives on dev1, the op on
      // dev0.
        Plan p;
        p.entry.push_back(eb(kDev1, 0, kA, 0, true));
        p.queue.push_back(make_tile_op(
            kT1, kDev0, {opnd(kA, 0, Access::Read, kDev1, 0)}));
        expect_fault(run_check(w, p, sequence),
            "operand-node-mismatch", CheckSite::Queue, 0, &p.queue[0]);
    }
    { // access-unlinked: a Read of an unlinked chunk.
        Plan p;
        p.queue.push_back(make_tile_op(
            kT1, kDev0, {opnd(kA, 0, Access::Read, kDev0, 0)}));
        expect_fault(run_check(w, p, sequence), "access-unlinked",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // access-unlinked: a Write too — "every operand, any access".
        Plan p;
        p.queue.push_back(make_tile_op(
            kT1, kDev0, {opnd(kA, 0, Access::Write, kDev0, 0)}));
        expect_fault(run_check(w, p, sequence), "access-unlinked",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // read-outdated: a Read of an outdated binding.
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, false));
        p.queue.push_back(make_tile_op(
            kT1, kDev0, {opnd(kA, 0, Access::Read, kDev0, 0)}));
        expect_fault(run_check(w, p, sequence), "read-outdated",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // read-outdated: a ReadWrite of an outdated binding.
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, false));
        p.queue.push_back(make_tile_op(
            kT1, kDev0, {opnd(kA, 0, Access::ReadWrite, kDev0, 0)}));
        expect_fault(run_check(w, p, sequence), "read-outdated",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // duplicate-operand: one (node, chunk) twice.
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, true));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Read, kDev0, 0),
                opnd(kA, 0, Access::Read, kDev0, 0)}));
        expect_fault(run_check(w, p, sequence), "duplicate-operand",
            CheckSite::Queue, 0, &p.queue[0]);
    }
    { // duplicate-write: two writers of one logical chunk on distinct
      // chunks of the op's own node. The fold checks LINKED, never the
      // operand's declared identity against the binding's (there is no
      // identity row), so chunk 1 may hold B while the operand
      // declares (A,0) — row 10 is over the DECLARED logical chunks.
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, false));
        p.entry.push_back(eb(kDev0, 1, kB, 0, false));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Write, kDev0, 0),
                opnd(kA, 0, Access::Write, kDev0, 1)}));
        expect_fault(run_check(w, p, sequence), "duplicate-write",
            CheckSite::Queue, 0, &p.queue[0]);
    }
}

// ---- the correspondence faults  ----

void correspondence_faults()
{
    World w;

    { // task-invented: TX is runnable (the fold passes row 5) but is
      // not of the task sequence — the finding is the op's own.
        Plan p;
        p.queue.push_back(make_link(kA, 0, kDev0, 0));
        p.queue.push_back(make_tile_op(kTX, kDev0,
            {opnd(kA, 0, Access::Write, kDev0, 0)}));
        std::vector<SeqTask> sequence{
            seq_task(kT1, {ta(kA, Access::Write)})};
        expect_fault(run_check(w, p, sequence), "task-invented",
            CheckSite::Queue, 1, &p.queue[1]);
    }
    { // task-missing: T2 of the sequence never runs; the finding is
      // the queue's close, the census carries the whole queue as the
      // ops prefix.
        Plan p;
        p.queue.push_back(make_link(kA, 0, kDev0, 0));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Write, kDev0, 0)}));
        std::vector<SeqTask> sequence{seq_task(kT1, {ta(kA, Access::Write)}),
            seq_task(kT2, {ta(kB, Access::Write)})};
        CheckResult r = run_check(w, p, sequence);
        expect_fault(r, "task-missing", CheckSite::Close, 0);
        CHECK(r.census->prefix.size() == p.queue.size());
        CHECK(!r.op.has_value());
    }
    { // task-order-violation: both tasks appear , the
      // runs come in the wrong order — the finding is the first
      // divergent run's first op.
        Plan p;
        p.queue.push_back(make_link(kA, 0, kDev0, 0));
        p.queue.push_back(make_link(kB, 0, kDev0, 1));
        p.queue.push_back(make_tile_op(kT2, kDev0,
            {opnd(kB, 0, Access::Write, kDev0, 1)}));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Write, kDev0, 0)}));
        std::vector<SeqTask> sequence{seq_task(kT1, {ta(kA, Access::Write)}),
            seq_task(kT2, {ta(kB, Access::Write)})};
        expect_fault(run_check(w, p, sequence), "task-order-violation",
            CheckSite::Queue, 2, &p.queue[2]);
    }
    { // task-order-violation: an extra run of T1 — one symbol too
      // many; a non-TILE_OP op between the runs keeps them two (only
      // TILE_OP runs collapse).
        Plan p;
        p.queue.push_back(make_link(kA, 0, kDev0, 0));
        p.queue.push_back(make_link(kB, 0, kDev0, 1));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Write, kDev0, 0)}));
        p.queue.push_back(make_unlink(kB, 0, kDev0, 1));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Write, kDev0, 0)}));
        std::vector<SeqTask> sequence{
            seq_task(kT1, {ta(kA, Access::Write)})};
        expect_fault(run_check(w, p, sequence), "task-order-violation",
            CheckSite::Queue, 4, &p.queue[4]);
    }
    { // operand-coverage-violation, the gap: W holds two chunks, the
      // task's single operand entry covers index 0 only — the census
      // names the task's first op as the witness and the whole queue
      // as the prefix.
        Plan p;
        p.queue.push_back(make_link(kW, 0, kDev0, 0));
        p.queue.push_back(make_link(kW, 1, kDev0, 1));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kW, 0, Access::Write, kDev0, 0)}));
        std::vector<SeqTask> sequence{
            seq_task(kT1, {ta(kW, Access::Write)})};
        CheckResult r = run_check(w, p, sequence);
        expect_fault(r, "operand-coverage-violation", CheckSite::Close, 2);
        CHECK(r.op.has_value());
        if (r.op.has_value())
        {
            CHECK(*r.op == p.queue[2]);
        }
        CHECK(r.census->prefix.size() == p.queue.size());
    }
    { // operand-coverage-violation, the outside entry: an operand
      // names (C,0), a tile of the table but not of the task's
      // operands — the finding is that entry's op.
        Plan p;
        p.entry.push_back(eb(kDev0, 1, kC, 0, true));
        p.queue.push_back(make_link(kA, 0, kDev0, 0));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Write, kDev0, 0),
                opnd(kC, 0, Access::Read, kDev0, 1)}));
        std::vector<SeqTask> sequence{
            seq_task(kT1, {ta(kA, Access::Write)})};
        expect_fault(run_check(w, p, sequence),
            "operand-coverage-violation", CheckSite::Queue, 1,
            &p.queue[1]);
    }
    { // access-mode-mismatch: the task reads A, and one operand entry
      // names a replica of (A,0) with Write — the fold passes (one
      // writer per logical chunk, the Read replica actual), coverage
      // collapses the replica reads to one covered chunk, and the check
      // catches the mode.
        Plan p;
        p.entry.push_back(eb(kDev0, 0, kA, 0, true));
        p.entry.push_back(eb(kDev0, 1, kB, 0, false));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Read, kDev0, 0),
                opnd(kA, 0, Access::Write, kDev0, 1)}));
        std::vector<SeqTask> sequence{
            seq_task(kT1, {ta(kA, Access::Read)})};
        expect_fault(run_check(w, p, sequence), "access-mode-mismatch",
            CheckSite::Queue, 0, &p.queue[0]);
    }
}

// ---- repeated TaskIds: one engine symbol, many sequence occurrences ----

// A workload shape: one TaskId at two
// sequence positions with OVERLAPPING tile sets and flipped roles —
// occurrence 0 writes A and reads B, occurrence 1 reads A and writes
// B. The correspondence runs over the TASK SEQUENCE, and the checks are
// per occurrence: each collapsed run accumulates
// against ITS OWN sequence entry's operand list, so the flipped
// accesses are two legal correspondences of one engine symbol and the
// plan passes — a merged per-TaskId access map censused exactly this
// plan as access-mode-mismatch at the second occurrence's op.
void repeated_task_occurrences()
{
    World w;
    std::vector<SeqTask> const sequence{
        seq_task(kT1, {ta(kA, Access::Write), ta(kB, Access::Read)}),
        seq_task(kT1, {ta(kA, Access::Read), ta(kB, Access::Write)})};

    { // The flip, both directions at once, PASSES with the exact
      // hand-computed exit.
        Plan p;
        p.entry.push_back(
            eb(kDev0, 1, kB, 0, true)); // B actual for occurrence 0's read
        p.queue.push_back(make_link(kA, 0, kDev0, 0));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Write, kDev0, 0),
                opnd(kB, 0, Access::Read, kDev0, 1)})); // occurrence 0
        p.queue.push_back(make_link(kC, 0, kDev0, 2));  // breaks the run
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Read, kDev0, 0), // occurrence 1 reads
                opnd(kB, 0, Access::Write, kDev0, 1)})); // what it wrote
        CheckResult r = run_check(w, p, sequence);
        CHECK(r.pass);
        // A actual (occurrence 0's write), B actual (the entry's flag,
        // re-actualized by occurrence 1's write), C outdated (its
        // link; no producer).
        std::vector<EntryBinding> expected{eb(kDev0, 0, kA, 0, true),
            eb(kDev0, 1, kB, 0, true), eb(kDev0, 2, kC, 0, false)};
        CHECK(r.exit.size() == expected.size());
        if (r.exit.size() == expected.size())
        {
            for (std::size_t i = 0; i < expected.size(); ++i)
            {
                CHECK(r.exit[i] == expected[i]);
            }
        }
    }
    { // The TRUE mismatch still fires — against the entry's OWN
      // occurrence: occurrence 1 reads A, its op carries the covering
      // Read plus a Write of A on a second chunk (linked, one writer
      // per logical chunk — legal to the fold), and the checker censuses
      // the Write with the census naming the op and the position.
        Plan p;
        p.entry.push_back(eb(kDev0, 1, kB, 0, true));
        p.queue.push_back(make_link(kA, 0, kDev0, 0));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Write, kDev0, 0),
                opnd(kB, 0, Access::Read, kDev0, 1)}));
        p.queue.push_back(make_link(kC, 0, kDev0, 2));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Read, kDev0, 0),
                opnd(kB, 0, Access::Write, kDev0, 1),
                opnd(kA, 0, Access::Write, kDev0, 2)}));
        expect_fault(run_check(w, p, sequence), "access-mode-mismatch",
            CheckSite::Queue, 3, &p.queue[3]);
    }
    { // A repeat whose second occurrence never ran: the TASK ran (the
      // per-task rule passes), the collapsed symbols form a strict
      // prefix — a task-order-violation, not task-missing.
        Plan p;
        p.queue.push_back(make_link(kA, 0, kDev0, 0));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Write, kDev0, 0)}));
        std::vector<SeqTask> const two{
            seq_task(kT1, {ta(kA, Access::Write)}),
            seq_task(kT1, {ta(kA, Access::Read)})};
        CheckResult r = run_check(w, p, two);
        expect_fault(r, "task-order-violation", CheckSite::Close, 0);
        CHECK(!r.op.has_value());
    }
    { // ADJACENT occurrences — no interleaved op splits the run (the
      // no-pressure pool's plan shape; the >=7716-MiB ladder's
      // deterministic census): the sequence itself names the task
      // again at the next entry, so the consecutive TILE_OP is that
      // NEXT occurrence — the run splits on the sequence's own
      // repeat, each occurrence accumulating against its own entry,
      // and the plan passes exactly as its interleaved twin above.
        Plan p;
        p.entry.push_back(
            eb(kDev0, 1, kB, 0, true)); // B actual for occurrence 0's read
        p.queue.push_back(make_link(kA, 0, kDev0, 0));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Write, kDev0, 0),
                opnd(kB, 0, Access::Read, kDev0, 1)})); // occurrence 0
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Read, kDev0, 0), // occurrence 1 reads
                opnd(kB, 0, Access::Write, kDev0, 1)})); // what it wrote
        CheckResult r = run_check(w, p, sequence);
        CHECK(r.pass);
        std::vector<EntryBinding> expected{eb(kDev0, 0, kA, 0, true),
            eb(kDev0, 1, kB, 0, true)};
        CHECK(r.exit == expected);
    }
    { // The adjacency split is the SEQUENCE's repeat, never a license
      // for a genuinely-single occurrence to double: one sequence
      // entry of T1, two consecutive T1 TILE_OPs with AGREEING
      // accesses — the run MERGES (the split demands the repeat at
      // the next entry), both TILE_OPs accumulate into the one
      // occurrence, and the plan passes as the split form always did.
        Plan p;
        p.queue.push_back(make_link(kA, 0, kDev0, 0));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Write, kDev0, 0)}));
        p.queue.push_back(make_tile_op(kT1, kDev0,
            {opnd(kA, 0, Access::Write, kDev0, 0)}));
        std::vector<SeqTask> const single{
            seq_task(kT1, {ta(kA, Access::Write)})};
        CheckResult r = run_check(w, p, single);
        CHECK(r.pass);
        std::vector<EntryBinding> expected{eb(kDev0, 0, kA, 0, true)};
        CHECK(r.exit == expected);
    }
}

// ---- the caller-defect surface: loud throws, never plan faults ----

void caller_defects()
{
    World w;
    Plan p;
    p.queue.push_back(make_link(kA, 0, kDev0, 0));
    std::vector<SeqTask> sequence{seq_task(kT1, {ta(kA, Access::Write)})};

    { // No plan.
        CheckInput in;
        in.inventory = &w.inv;
        in.tiles = &w.tiles;
        in.runnable = &w.runnable;
        in.task_sequence = &sequence;
        CHECK(throws_loud([&] { (void) check(in); }, "no plan"));
    }
    { // No inventory.
        CheckInput in;
        in.plan = &p;
        in.tiles = &w.tiles;
        in.runnable = &w.runnable;
        in.task_sequence = &sequence;
        CHECK(throws_loud([&] { (void) check(in); }, "no inventory"));
    }
    { // No tile table.
        CheckInput in;
        in.plan = &p;
        in.inventory = &w.inv;
        in.runnable = &w.runnable;
        in.task_sequence = &sequence;
        CHECK(throws_loud([&] { (void) check(in); }, "no tiles"));
    }
    { // No executability table.
        CheckInput in;
        in.plan = &p;
        in.inventory = &w.inv;
        in.tiles = &w.tiles;
        in.task_sequence = &sequence;
        CHECK(throws_loud([&] { (void) check(in); }, "no runnable"));
    }
    { // No task sequence.
        CheckInput in;
        in.plan = &p;
        in.inventory = &w.inv;
        in.tiles = &w.tiles;
        in.runnable = &w.runnable;
        CHECK(
            throws_loud([&] { (void) check(in); }, "no task_sequence"));
    }
    { // A task sequence naming a tile the tile table does not carry:
      // a registry defect, loud.
        std::vector<SeqTask> broken{seq_task(kT1, {ta(77, Access::Write)})};
        CheckResult r;
        bool threw = false;
        try
        {
            r = run_check(w, p, broken);
        }
        catch (std::runtime_error const &e)
        {
            threw = std::string(e.what()).find("tile 77") !=
                std::string::npos;
        }
        CHECK(threw);
    }
}

// ---- the tile table and the chunk span : the
//      registered payload bytes survive beside the chunk count, and
//      the occupied span of every chunk index is min(chunk_class_bytes(),
//      nbytes - index * chunk_class_bytes()) — a full chunk the whole
//      class, the tail chunk its payload remainder and nothing more,
//      an index past the decomposition zero. The registered bytes are
//      the fact the executor's dispatch and the tracer's byte stamp
//      derive a COPY's transferred span from.

void tile_table_and_span()
{
    TileTable tiles;

    // The rows: one tile an exact multiple of the class (2 classes),
    // one tail tile (2 classes plus a quarter-class remainder), one
    // exactly-one-class tile.
    TileId const kTwo = 0;
    TileId const kTail = 1;
    TileId const kOne = 2;
    tiles.add(kTwo, 2 * chunk_class_bytes());
    tiles.add(kTail, 2 * chunk_class_bytes() + chunk_class_bytes() / 4);
    tiles.add(kOne, chunk_class_bytes());

    // The counts and the bytes — both facts of the one registration,
    // never stored apart.
    CHECK(tiles.carries(kTwo) && tiles.carries(kTail));
    CHECK(tiles.carries(kOne));
    CHECK(tiles.n(kTwo) == 2);
    CHECK(tiles.n(kTail) == 3);
    CHECK(tiles.n(kOne) == 1);
    CHECK(tiles.nbytes(kTwo) == 2 * chunk_class_bytes());
    CHECK(tiles.nbytes(kTail) ==
        2 * chunk_class_bytes() + chunk_class_bytes() / 4);
    CHECK(tiles.nbytes(kOne) == chunk_class_bytes());

    // The chunk-span rule over EVERY chunk index of the exact-multiple tile:
    // full chunks the whole class, past the decomposition zero.
    for (ChunkIndex i = 0; i < 4; ++i)
    {
        std::size_t const off = i * chunk_class_bytes();
        std::size_t const expect =
            off < 2 * chunk_class_bytes() ? chunk_class_bytes() : 0;
        CHECK(tile_chunk_span(2 * chunk_class_bytes(), i) == expect);
    }

    // The tail tile: two full chunks, then the quarter-class tail,
    // then nothing — the wasted bytes of the tail's block belong to no
    // span.
    std::size_t const kTailBytes =
        2 * chunk_class_bytes() + chunk_class_bytes() / 4;
    CHECK(tile_chunk_span(kTailBytes, 0) == chunk_class_bytes());
    CHECK(tile_chunk_span(kTailBytes, 1) == chunk_class_bytes());
    CHECK(tile_chunk_span(kTailBytes, 2) == chunk_class_bytes() / 4);
    CHECK(tile_chunk_span(kTailBytes, 3) == 0);

    // The through-the-table form (the executor's and the tracer's own
    // derivation): tile_chunk_span(tiles.nbytes(t), index).
    CHECK(tile_chunk_span(tiles.nbytes(kTail), 2) ==
        chunk_class_bytes() / 4);
    CHECK(tile_chunk_span(tiles.nbytes(kOne), 0) == chunk_class_bytes());

    // The zero-byte tile: the degenerate VALID case  —
    // n = 0, no index exists, every span 0.
    tiles.add(9, 0);
    CHECK(tiles.n(9) == 0);
    CHECK(tile_chunk_span(tiles.nbytes(9), 0) == 0);

    // The loud belts: a duplicate registration, an uncarried tile's
    // lookup — never serviced, loud with the fault's own name.
    CHECK(throws_loud([&] { tiles.add(kOne, chunk_class_bytes()); },
        "duplicate tile"));
    CHECK(throws_loud([&] { (void) tiles.n(77); }, "unknown-tile"));
    CHECK(throws_loud(
        [&] { (void) tiles.nbytes(77); }, "unknown-tile"));
}

} // namespace

int main()
{
    positive_full_plan();
    positive_chained();
    malformed_plan_variants();
    unknown_id_rows();
    unknown_tile_rows();
    index_out_of_range_rows();
    linkage_rows();
    copy_rows();
    tileop_rows();
    correspondence_faults();
    repeated_task_occurrences();
    caller_defects();
    tile_table_and_span();
    if (g_checks_failed != 0)
    {
        std::cerr << g_checks_failed << " check(s) failed\n";
        return 1;
    }
    std::cout << "plan_checker: all checks passed\n";
    return 0;
}
