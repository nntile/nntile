/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * stateless plan checker of the chunk-graph executor: data in, verdict
 * and products out — one pure pass, the one fold, always full.
 *
 * @file src/plan_checker.hh
 * The checker. CheckInput carries exactly the five mandatory inputs
 * plus the optional chain; check() runs the entry-state check before
 * op 0, the chaining check whenever a chain input is supplied, the
 * fold over the queue through the four per-op dispatch tables (every
 * row in its fixed group order — shape, references, identity/linkage,
 * topology, flags, within-op distinctness — every fault name verbatim,
 * all rows before any effect), and the correspondence scan over the
 * same walk. CheckResult is the verdict — pass or fail, the fault's
 * name, the position, the offending op verbatim, the census, and on
 * pass the exit state. There are no check levels and no certification:
 * one check, always full. Data only, never observations — the duration
 * model, the device, and events are not inputs — and the checker edits
 * nothing: the plan that leaves a passed check is the plan that
 * entered it, byte for byte; verdict and exit state travel beside the
 * plan. No plan reaches execute mode without a passed check — the
 * caller enforces it; the checker reports. Complexity: one pass,
 * O(ops + operands + C + entry), C the task sequence's operand-chunk
 * total — no sort, no rescan.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include "plan.hh"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nnhaul
{

// ---- the input  ----

// Data only, never observations; the checker edits nothing. The five
// mandatory inputs plus the optional chain, present whenever the
// executing pipeline supplies the predecessor's exit fold; without it,
// a declared cold start, the entry state stands on its own rows.
struct CheckInput
{
    Plan const *plan = nullptr; // 1. entry state + queue
    NodeInventory const *inventory = nullptr; // 2. cardinalities +
                                              //    the channel matrix
    TileTable const *tiles = nullptr;         // 3. TileId -> chunk count n
    ExecutabilityTable const *runnable =
        nullptr; // 4. task -> runnable nodes
    std::vector<SeqTask> const *task_sequence =
        nullptr; // 5. the batch's tasks, topo order, operand tiles
                 //    + their accesses
    std::optional<std::vector<EntryBinding>> chain; // the predecessor's
                                                    // exit fold
};

// ---- the verdict: fault name, position, op, census, exit ----

// Where a finding sits. Entry findings (the entry-state rows and the
// chaining check) precede op 0 — position then names the offending
// ENTRY BINDING's index (0 for the chaining check, which offends
// nowhere in particular); Queue findings are the fold's rows and the
// correspondence rows that name an op — position is that op's queue
// index; Close findings are the end-of-queue correspondence rows —
// position carries the census witness's queue index when one exists.
enum class CheckSite : std::uint8_t
{
    Entry,
    Queue,
    Close
};

// Pass/fail, the finding, and the products beside the plan. On pass:
// exit is the fold's snapshot and fault, op, and census are
// empty/absent — the census is the loud-failure artifact and never
// rides a passing plan. On fail: fault is the fault's NAME verbatim,
// site/position the place, op the offending op verbatim when one
// exists, and census the diagnostic snapshot — the fold rendered per
// node with the free lists beside it, the entry state, the ops prefix,
// the inputs' relevant facts.
struct CheckResult
{
    bool pass = false;
    std::string fault;                 // the fault's name; empty on pass
    CheckSite site = CheckSite::Entry; // meaningless on pass
    std::size_t position = 0;
    std::optional<PlanOp> op;          // the offending op verbatim
    std::optional<Census> census;      // on fail: the diagnostic snapshot
    std::vector<EntryBinding> exit;    // on pass: the exit state
};

// ---- the check (stateless, one pure pass) ----

// The entry-state check, the chaining check when the chain is
// supplied, the fold with its per-op precondition tables, and the
// correspondence scan — one pass, always full. Throws
// std::runtime_error only on a malformed INPUT — a null mandatory
// pointer, or a task sequence naming a tile the tile table does not
// carry (a registry defect) — a caller defect, never a plan defect: a
// plan defect is the returned verdict, census and all, never a throw.
CheckResult check(CheckInput const &input);

} // namespace nnhaul
