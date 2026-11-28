/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * timing pass: pure arithmetic over the queue, the folded owed sets,
 * the duration model, and the timing boundary.
 *
 * @file src/timing.hh
 * The simulator of the plan: the lanes ARE the servers, one in-order
 * FIFO clock per LaneId, run-absolute, zero at cold start and
 * continuous across plan boundaries; the start rule is the maximum of
 * the own lane clock and every folded owed op's simulated end; the end
 * rule is start plus the modeled duration; LINK and UNLINK contribute
 * nothing. The products (the simulated trace, the per-plan local
 * makespans, the run makespan, the swimlane frames) travel BESIDE the
 * plan, never plan fields and never inputs: no function here writes
 * into a Plan, and no planning or execution function may consume what
 * this file returns. A duration-model lookup that misses aborts the
 * pass loudly with a std::runtime_error naming duration-model-miss,
 * never serviced.
 *
 * The owed-sets seam (pinned for src/dependency.hh, a sibling piece):
 * this pass consumes the SAME folded owed set the executor issues, as a
 * plain view, std::vector<std::vector<OpRef>> const &owed_sets — one
 * entry per queue position, the FOLDED set, empty where an op owes
 * nothing; LINK/UNLINK entries are never consulted. The run-scoped
 * owed-end registry (OpRef -> simulated end) is retired by the SAME
 * displacement rule as the event registry: an entry survives while a
 * fact of the current dependency state names its op, and NOTHING else
 * retires anything. Because the dependency state lives in
 * dependency.hh, this file takes retirement as INPUT: the caller (the
 * engine, at wave close) derives the live refs from the dependency
 * state's facts and hands them to retire(), which drops every owed_end
 * entry not named. Retirement is hygiene, never arithmetic — a correct
 * live set changes not one number (a displaced fact is never named
 * again) — so the step-wise TimingPass and the one-shot simulate_run
 * below agree whatever retirement schedule the caller keeps.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include "plan.hh"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nnhaul
{

// One simulated op — queue position, kind, LANE, simulated start,
// simulated end. The trace carries no plan_seq (plans concatenate in
// issue order) and LINK/UNLINK never appear. The _ns suffix names the
// model's own denomination (nanoseconds); the pass moves plain uint64
// ticks and converts nothing — the unit is the configuration's choice.
struct SimTraceRecord
{
    std::size_t op_position;
    PlanOpKind kind;
    LaneId lane;
    std::uint64_t start_ns, end_ns;
};

// One plan's products, returned BESIDE the plan: the
// LOCAL makespan — the maximum simulated end over the plan's OWN ops —
// and the plan's slice of the run trace, records in queue order.
struct PlanTiming
{
    std::uint64_t local_makespan_ns = 0;
    std::vector<SimTraceRecord> records;
};

// The run's products: the per-plan timings in issue order, the whole
// run trace (the plans' slices concatenated), and the RUN makespan —
// the maximum over ALL lane clocks, untouched lanes included: the
// run's predicted end-to-end, never a stand-in for a local makespan
// nor the reverse.
struct RunTiming
{
    std::vector<PlanTiming> plans;
    std::vector<SimTraceRecord> trace;
    std::uint64_t run_makespan_ns = 0;
};

// ---- the swimlane frames  ----

// The render frame: Full — one row per lane of the inventory, the
// DEFAULT (an empty lane is itself a finding: a node or channel the
// planner left idle); TouchedSoFar — the one concession, the monotone
// union of every lane an op has landed on since cold start, never a
// per-plan subset.
enum class LaneFrame : std::uint8_t { Full, TouchedSoFar };

// One lane's row of the structured record: the lane and its segments
// — the (position, kind, start, end) records on it, in queue order,
// which is lane order for an in-order lane.
struct SimLaneSegments
{
    LaneId lane = kNoLane;
    std::vector<SimTraceRecord> segments;
};

// The structured swimlane record: one row per lane of the frame, in
// ascending LaneId order. The text render is built from this record
// and names lanes via NodeInventory::lane_name — exec(2), chan(0,3)
// style; NEVER vendors or runtimes.
struct SimSwimlanes
{
    LaneFrame frame = LaneFrame::Full;
    std::vector<SimLaneSegments> lanes;
};

// The text render of a structured record: one line per lane of the
// frame, each segment as [p<position> <kind> <start>..<end], an empty
// lane rendered "(idle)" — the finding the full frame exists to show.
std::string render_swimlanes(
    SimSwimlanes const &lanes, NodeInventory const &inv);

// ---- the timing pass  ----

// Run-scoped, like the lane clocks and the registries: constructed at
// cold start, stepped once per plan in issue order. The duration model
// and the node inventory are read-only references — INPUT only, never
// an observation; the pass queries no device, reads no event, consults
// no runtime fact. The engine never reorders through this pass and the
// pass never reorders anything itself: it is arithmetic, and what a
// caller does with comparable candidate plans — always on the same
// timing boundary — is the caller's act.
class TimingPass
{
  public:
    TimingPass(NodeInventory const &inv, DurationModel const &model);

    // One plan's step: the folded owed sets per queue position — at
    // most one entry per op; a missing entry is the empty set; an owed
    // ref naming an op with no registered simulated end is the loud
    // owed-end-miss (the registry-miss analog — a caller defect, never
    // serviced). Plans must step in strictly increasing plan_seq (issue
    // order): OpRefs pack the seq, and two plans of one seq would
    // collide in the owed-end registry — loud, never a silent
    // overwrite.
    PlanTiming run_plan(Plan const &plan,
        std::vector<std::vector<OpRef>> const &owed_sets);

    // Retirement by displacement: drop every owed_end entry not named
    // by live_refs. The caller derives live_refs from the dependency
    // state — every op a current fact names (a USER fact or a PRODUCER
    // fact; the USER fact never resets). A live ref with no registry
    // entry is vacuous (already retired, or never timed), not a fault.
    void retire(std::vector<OpRef> const &live_refs);

    // The RUN makespan: the maximum over ALL lane clocks, untouched
    // lanes included — live at any step, the last simulate's run
    // makespan of the stats surface.
    std::uint64_t run_makespan() const;

    // The whole run trace so far, plans concatenated in issue order.
    std::vector<SimTraceRecord> const &trace() const;

    // The structured swimlane record over the frame.
    SimSwimlanes swimlanes(LaneFrame frame) const;

    // One text render, explicit frame, the full inventory the default.
    std::string render_swimlanes(LaneFrame frame = LaneFrame::Full) const;

  private:
    NodeInventory const *inv_;
    DurationModel const *model_;
    std::vector<std::uint64_t> lane_clock_; // run-absolute; zero at
                                            // cold start
    std::vector<bool> touched_;             // the monotone union since
                                            // cold start
    std::unordered_map<OpRef, std::uint64_t> owed_end_;
    std::vector<SimTraceRecord> trace_;
    std::uint64_t last_plan_seq_ = 0;
    bool have_plan_ = false;
};

// The one-shot entry: the run's plans in issue order, each plan's
// folded owed sets (owed_sets[i] is plan i's
// per-position view; a missing plan entry is the all-empty view), and
// the retirement seam as a callback — live_refs(i), invoked after plan
// i, returns the ops the dependency state still names; a null callback
// retires nothing (correct arithmetic, unbounded registry). Returns
// the per-plan timings, the run trace, and the run makespan.
RunTiming simulate_run(std::vector<Plan> const &plans,
    NodeInventory const &inv, DurationModel const &model,
    std::vector<std::vector<std::vector<OpRef>>> const &owed_sets,
    std::function<std::vector<OpRef>(std::size_t)> live_refs = {});

} // namespace nnhaul
