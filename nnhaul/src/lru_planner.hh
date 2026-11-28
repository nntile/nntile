/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * lowering walk of the chunk-graph executor: the planner that turns
 * one batch's task sequence into a plan of exactly the four op kinds,
 * by folding its own ops over a working copy of the binding table.
 *
 * @file src/lru_planner.hh
 * The LRU lowering planner: the lowering walk emits the four op kinds
 * over the plan-time state read, and pool exhaustion is censused at
 * the read. Every plan-time query — free lists, a logical chunk's
 * bindings and flags, pool cardinalities — IS answered by a
 * BindingTable working copy seeded at the entry state and advanced by
 * the SAME transitions as the ops appended, so the exit state IS the
 * predicted fold, never a live device or live table; no observation
 * participates in planning. Scheduling lives elsewhere: the task's
 * node is an INPUT, queue order IS task order, and within a task the
 * operands resolve in the order given with chunk indexes ascending —
 * the planner's one lowering act, fixed to the whole-tile form, never
 * a scheduling decision. Victim selection is LRU by last planned touch;
 * the victim-selection ORDER is fixed and MUST NOT change (one policy,
 * LRU) — with ONE liveness class ahead of it: the bindings of DEAD
 * tiles whose last use has passed are immediate victims (freed bare;
 * liveness, not recency policy — a live tile's victim order is LRU,
 * unchanged). Pool exhaustion is the plan-time fault, emitted at the
 * state read with a census and never serviced. The queue carries
 * exactly Link, Unlink, Copy, TileOp and nothing else.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include "binding_table.hh"
#include "plan_checker.hh"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nnhaul
{

// ---- the product: the plan plus the predicted exit, beside it ----

// One close-and-plan's product: the two-part Plan (entry
// state plus queue) and, RETURNED BESIDE IT, never a plan field, the
// predicted exit state — snapshot() of the working copy after folding
// the whole queue, the successor plan's seed — plus the walk's
// measurement counts (the same state the engine's counters read;
// beside the product like the exit fold, never in the queue).
struct PlannedBatch
{
    Plan plan;
    std::vector<EntryBinding> exit; // the predicted exit fold

    // The walk's own counts, one per plan (the engine aggregates them
    // into its counters): evictions (victims taken), rescues (bytes
    // landed elsewhere to survive), refetches (actual-data deliveries
    // into an outdated binding at the consumer's own resolution — the
    // just-in-time path), prefetches (early refetch fires through the
    // slack gate — the same delivery emitted at a free-list gain the
    // horizon proves spare, so the copy overlaps the compute between
    // the fire and the consumer), dying frees (bare unlinks of dead
    // tiles at their dying positions), and the entry/exit binding
    // counts (the residency the walk started and ended with).
    struct Stats
    {
        std::uint64_t evictions = 0;
        std::uint64_t rescues = 0;
        std::uint64_t refetches = 0;
        std::uint64_t prefetches = 0;
        std::uint64_t dying_frees = 0;
        std::uint64_t entry_bindings = 0;
        std::uint64_t exit_bindings = 0;
        // The fragmentation meter (diagnosis only): multi-chunk
        // contiguous-handout misses, the largest chain that missed,
        // and the longest free run seen at any miss.
        std::uint64_t run_misses = 0;
        std::uint32_t run_miss_want = 0;
        std::size_t run_miss_longest = 0;
        // Exhaustion pin spends (diagnosis only: how often a pinned
        // chunk had to be spent — never a gate).
        std::uint64_t pin_spends = 0;
    };
    Stats stats;
};

// ---- the plan-time census throw ----

// The loud plan-time failure: pool exhaustion aborts the walk — no
// retry, no repair, no replan, a plan never ships incomplete. The
// census payload holds the fault's name and the need (the node whose
// free list is empty, the task/tile-chunk that needed the binding),
// the fold snapshot (the working copy rendered, every node's free
// list printed beside it), the entry state, the ops prefix built so
// far, and the inputs' relevant facts rendered. It travels beside the
// plan, never in it.
struct PlanCensusError : std::runtime_error
{
    Census census;

    PlanCensusError(std::string const &reason, Census payload) :
        std::runtime_error(reason), census(std::move(payload))
    {
    }
};

// ---- the checker's task sequence, from the request ----

// The conversion surface for the checker and the engine: the request's
// PlannedTask list IS the SeqTask source — the batch's tasks in the
// tile graph's topological order, each with its operand tiles and
// their accesses. One TileAccess per TileUse, order preserved.
std::vector<SeqTask> to_seq_tasks(
    std::vector<PlannedTask> const &tasks);

// ---- the lowering walk  ----

// Stateless per call: the whole planner state is the walk's own
// session (the working copy, the recency orders, the queue under
// construction), built fresh from the request and discarded into the
// product — nothing persists across calls, the successor plan's seed
// being the returned exit fold alone.
class LruPlanner
{
  public:
    // Build the batch's plan over the plan-time state read.
    // Deterministic: the same request, inventory, and tile table
    // produce the same queue op for op (the pinned orders below).
    // Caller defects throw std::runtime_error loud, never serviced:
    // an unknown task node, a tile the table does not carry, a
    // duplicate tile in one task (a repeated TaskId in the BATCH is
    // legal — the correspondence collapses each maximal run of
    // consecutive TILE_OPs of one task into one symbol, and the
    // engine's multiplied task symbols map many instances onto one
    // symbol), an entry state not sorted ascending by (node, chunk) or
    // naming one chunk twice, a read of a logical chunk no actual
    // binding anywhere can feed (data the predicted world never
    // produced). The one plan-time FAULT is pool exhaustion, thrown as
    // PlanCensusError with the census, never serviced.
    //
    // `validate` is a TEST convenience only, a plain bool and no env
    // knob: when true the walk runs plan_checker::check on its own
    // product — the task sequence via to_seq_tasks() and an
    // executability table stubbed permissively from the request (each
    // task runnable exactly on its named node; executability is the
    // registration piece's registry fact, outside the planner's
    // inputs) — and demands the pass and the checker's exit fold equal
    // the working copy's (the same function run twice). The ENGINE runs
    // the real check itself — no plan reaches execute without a passed
    // check — so the default is false.
    PlannedBatch plan(PlanRequest const &request,
        NodeInventory const &inventory,
        TileTable const &tiles,
        bool validate = false);
};

} // namespace nnhaul
