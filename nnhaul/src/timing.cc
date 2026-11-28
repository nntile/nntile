/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * timing pass's arithmetic: the lane clocks, the start and end rules,
 * the owed-end registry and its displacement retirement, the census-
 * style duration-model-miss, and the swimlane frames.
 *
 * @file src/timing.cc
 * The acts behind src/timing.hh: TimingPass's construction at cold
 * start (every lane clock zero), run_plan's per-op walk — lane_of_op,
 * the modeled duration (carries() first, the model's throwing lookup
 * the belt), the start rule's maximum over the lane clock and every
 * folded owed end, the end rule, the trace append, the local makespan —
 * retire()'s displacement drop, the run makespan over ALL lane clocks,
 * and the swimlane record and text render over the Full and TouchedSoFar
 * frames. Pure arithmetic: no device, no event, no runtime fact, no
 * scheduling, nothing written into a Plan.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "timing.hh"

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace nnhaul
{

namespace
{

char const *kind_name(PlanOpKind k)
{
    switch (k)
    {
    case PlanOpKind::Link:
        return "link";
    case PlanOpKind::Unlink:
        return "unlink";
    case PlanOpKind::Copy:
        return "copy";
    case PlanOpKind::TileOp:
        return "tile_op";
    }
    return "?";
}

std::string where(std::uint64_t plan_seq, std::size_t position)
{
    return "plan " + std::to_string(plan_seq) + " position " +
        std::to_string(position);
}

std::string ref_decode(OpRef r)
{
    return "(plan " + std::to_string(r >> 32) + ", position " +
        std::to_string(r & 0xffffffffu) + ")";
}

// The modeled duration of one lane-carrying op: D(task, node) for a
// TILE_OP, chunk_class_bytes() / R for a COPY over its channel lane.
// Completeness is checked up front: carries() first, then a
// duration-model-miss — the model's own throwing lookup is the belt
// (plan.cc), never the serviced path.
std::uint64_t modeled_duration(PlanOp const &op, LaneId lane,
    NodeInventory const &inv, DurationModel const &model,
    std::uint64_t plan_seq, std::size_t position)
{
    if (op.kind == PlanOpKind::TileOp)
    {
        TaskId const t = op.u.tile_op.task;
        NodeId const n = op.u.tile_op.node;
        if (!model.carries(t, n))
        {
            throw std::runtime_error(
                "duration-model-miss: D(task " + std::to_string(t) +
                ", node " + std::to_string(n) + ") at " +
                where(plan_seq, position) + " - the duration model "
                "carries no (task, node) entry");
        }
        return model.d_ns(t, n);
    }
    // PlanOpKind::Copy: the only other lane-carrying kind.
    if (!model.carries_channel(lane))
    {
        throw std::runtime_error("duration-model-miss: rate of " +
            inv.lane_name(lane) + " at " + where(plan_seq, position) +
            " - the duration model carries no channel rate");
    }
    return model.copy_ns(lane);
}

} // namespace

// ---- the timing pass  ----

TimingPass::TimingPass(NodeInventory const &inv, DurationModel const &model)
    : inv_(&inv), model_(&model)
{
    // Cold start: every lane clock zero, run-absolute; the touched
    // union empty. The lane set is inventory truth, static for the run.
    lane_clock_.assign(inv.lane_count(), 0);
    touched_.assign(inv.lane_count(), false);
}

PlanTiming TimingPass::run_plan(Plan const &plan,
    std::vector<std::vector<OpRef>> const &owed_sets)
{
    // Issue order: plan_seq packs into every OpRef, so a repeated or
    // regressed seq would collide entries of the owed-end registry -
    // loud, never a silent overwrite.
    if (have_plan_ && plan.plan_seq <= last_plan_seq_)
    {
        throw std::runtime_error("timing pass: plan_seq " +
            std::to_string(plan.plan_seq) + " does not follow plan_seq " +
            std::to_string(last_plan_seq_) +
            " - plans step in run-queue order");
    }
    if (owed_sets.size() > plan.queue.size())
    {
        throw std::runtime_error("timing pass: " +
            std::to_string(owed_sets.size()) + " owed sets for a queue of " +
            std::to_string(plan.queue.size()) +
            " ops - at most one folded owed set per queue position");
    }

    PlanTiming out;
    for (std::size_t p = 0; p < plan.queue.size(); ++p)
    {
        PlanOp const &op = plan.queue[p];
        LaneId const l = lane_of_op(op, *inv_);
        if (l == kNoLane)
        {
            if (op.kind == PlanOpKind::Copy)
            {
                // The checker's row, caught at check time in the
                // engine (simulate IS check plus this pass); skipping a
                // COPY here would hide it - loud, the belt behind the
                // check.
                throw std::runtime_error("copy-no-channel: COPY at " +
                    where(plan.plan_seq, p) + " over (node " +
                    std::to_string(op.u.copy.src.node) + ", chunk " +
                    std::to_string(op.u.copy.src.chunk) + ") -> (node " +
                    std::to_string(op.u.copy.dst.node) + ", chunk " +
                    std::to_string(op.u.copy.dst.chunk) +
                    ") - the inventory carries no such channel");
            }
            // LINK/UNLINK: no lane, no server, no times, never owed,
            // never minted - not in the trace.
            continue;
        }

        std::uint64_t const d =
            modeled_duration(op, l, *inv_, *model_, plan.plan_seq, p);

        // The start rule: the maximum of the own lane clock and every
        // folded owed op's simulated end; a missing term is skipped -
        // the maximum runs over existing terms only. An owed ref with
        // no registered end is a caller defect (a correct derivation
        // names only timed ops of the live dependency state), the
        // registry-miss analog - loud.
        std::uint64_t start = lane_clock_[l];
        if (p < owed_sets.size())
        {
            for (OpRef a : owed_sets[p])
            {
                auto it = owed_end_.find(a);
                if (it == owed_end_.end())
                {
                    throw std::runtime_error("owed-end-miss: the op at " +
                        where(plan.plan_seq, p) + " owes ref " +
                        ref_decode(a) +
                        " - no simulated end is registered (retired by "
                        "displacement, or never timed; the registry-miss "
                        "analog)");
                }
                if (it->second > start)
                {
                    start = it->second;
                }
            }
        }

        // The end rule, then the registry acts: the lane clock advances
        // and the op's end registers — every timed op registers;
        // minting is not the simulator's business (the executor prices
        // the event, the pass prices the arithmetic).
        std::uint64_t const end = start + d;
        lane_clock_[l] = end;
        owed_end_[make_op_ref(plan.plan_seq, p)] = end;
        touched_[l] = true;

        SimTraceRecord const rec{p, op.kind, l, start, end};
        trace_.push_back(rec);
        out.records.push_back(rec);
        if (end > out.local_makespan_ns)
        {
            out.local_makespan_ns = end; // the LOCAL makespan: the max
                                         // over this plan's OWN ops
        }
    }

    last_plan_seq_ = plan.plan_seq;
    have_plan_ = true;
    // The owed-end retirement after the plan is the CALLER's act (the
    // seam, src/timing.hh): retire(live_refs) with the live set the
    // dependency state names.
    return out;
}

void TimingPass::retire(std::vector<OpRef> const &live_refs)
{
    // Displacement only: an entry survives while a fact of the current
    // dependency state names its op; nothing else retires anything, and
    // an entry whose facts stand forever stays until the run ends.
    // live_refs comes from the dependency state (src/dependency.hh, the
    // sibling piece); a live ref with no registry entry is vacuous, not
    // a fault.
    if (live_refs.empty())
    {
        owed_end_.clear();
        return;
    }
    std::unordered_set<OpRef> live(live_refs.begin(), live_refs.end());
    for (auto it = owed_end_.begin(); it != owed_end_.end();)
    {
        if (live.find(it->first) == live.end())
        {
            it = owed_end_.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

std::uint64_t TimingPass::run_makespan() const
{
    // The RUN makespan: the maximum over ALL lane clocks, untouched
    // lanes included - an unowed tail on one lane outliving the
    // successor plan is exactly what a per-plan number would hide.
    std::uint64_t m = 0;
    for (std::uint64_t c : lane_clock_)
    {
        if (c > m)
        {
            m = c;
        }
    }
    return m;
}

std::vector<SimTraceRecord> const &TimingPass::trace() const
{
    return trace_;
}

// ---- the swimlane frames  ----

SimSwimlanes TimingPass::swimlanes(LaneFrame frame) const
{
    SimSwimlanes out;
    out.frame = frame;
    for (std::size_t l = 0; l < lane_clock_.size(); ++l)
    {
        if (frame == LaneFrame::TouchedSoFar && !touched_[l])
        {
            // The concession: the monotone touched-so-far
            // union since cold start - never a per-plan subset, so a
            // lane keeps its row once touched, whatever later plans do.
            continue;
        }
        SimLaneSegments row;
        row.lane = static_cast<LaneId>(l);
        for (SimTraceRecord const &rec : trace_)
        {
            if (rec.lane == row.lane)
            {
                // Queue order is lane order for an in-order lane,
                // so the filter preserves order.
                row.segments.push_back(rec);
            }
        }
        out.lanes.push_back(std::move(row));
    }
    return out;
}

std::string TimingPass::render_swimlanes(LaneFrame frame) const
{
    return nnhaul::render_swimlanes(swimlanes(frame), *inv_);
}

std::string render_swimlanes(
    SimSwimlanes const &lanes, NodeInventory const &inv)
{
    // Lanes, never vendors or runtimes : every name comes
    // from lane_name; an empty lane renders "(idle)" - the finding the
    // full frame exists to show.
    std::string out;
    for (SimLaneSegments const &row : lanes.lanes)
    {
        out += inv.lane_name(row.lane);
        if (row.segments.empty())
        {
            out += "  (idle)";
        }
        else
        {
            for (SimTraceRecord const &seg : row.segments)
            {
                out += "  [p" + std::to_string(seg.op_position) + " " +
                    kind_name(seg.kind) + " " +
                    std::to_string(seg.start_ns) + ".." +
                    std::to_string(seg.end_ns) + "]";
            }
        }
        out += "\n";
    }
    return out;
}

// ---- the one-shot entry  ----

RunTiming simulate_run(std::vector<Plan> const &plans,
    NodeInventory const &inv, DurationModel const &model,
    std::vector<std::vector<std::vector<OpRef>>> const &owed_sets,
    std::function<std::vector<OpRef>(std::size_t)> live_refs)
{
    if (owed_sets.size() > plans.size())
    {
        throw std::runtime_error("timing pass: " +
            std::to_string(owed_sets.size()) + " per-plan owed-set views "
            "for " + std::to_string(plans.size()) +
            " plans - at most one per plan");
    }

    TimingPass pass(inv, model);
    RunTiming out;
    for (std::size_t i = 0; i < plans.size(); ++i)
    {
        std::vector<std::vector<OpRef>> const empty;
        std::vector<std::vector<OpRef>> const &view =
            i < owed_sets.size() ? owed_sets[i] : empty;
        out.plans.push_back(pass.run_plan(plans[i], view));
        if (live_refs)
        {
            // The retirement seam as the callback form: the caller's
            // live set per plan, the same act as the step-wise retire()
            // after each plan.
            pass.retire(live_refs(i));
        }
    }
    out.trace = pass.trace();
    out.run_makespan_ns = pass.run_makespan();
    return out;
}

} // namespace nnhaul
