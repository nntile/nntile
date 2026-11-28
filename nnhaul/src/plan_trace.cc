/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * chunk-graph executor's tracer over the lane set: the measured
 * records' composition from the stamp facts, the measured swimlane
 * frames, the join against the simulated trace, and the Chrome Trace
 * Event dump.
 *
 * @file src/plan_trace.cc
 * The acts behind src/plan_trace.hh: PlanTrace's arming, the plan
 * registration deriving every op's lane (lane_of_op — the same pure
 * function the executor and the simulator walk), the queue-close
 * harvest composing the measured records from the stamp facts the
 * executor read off the lanes (each fact an engine-thread arrival after
 * the barriers; a missing pair or an inverted one is the loud
 * stamp-fidelity census), the measured records and their swimlane
 * framing beside the simulated side's rules, the (plan_seq, position)
 * join asserting lane-for-lane equality with the times as facts, and
 * the JSON dump whose every ts and dur derive from the stamps alone
 * (copy events carrying the occupied byte span beside them — the true
 * transferred bytes every direction's byte sum reads). Stamps never
 * gate anything; nothing here observes a device, an event, or a lane
 * mid-flight — the facts arrive only at the queue-close join, and every
 * method here runs on the engine thread.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "plan_trace.hh"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace nnhaul
{

namespace
{

// NNHAUL_VERSION arrives as a PRIVATE compile definition from
// CMakeLists.txt; the fallback covers sources compiled outside this
// build.
#ifdef NNHAUL_VERSION
std::string const kRuntimeVersion = NNHAUL_VERSION;
#else
std::string const kRuntimeVersion = "unknown";
#endif

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

// JSON string escaping: quotes, backslash, and the C0 controls; the
// rest (UTF-8 bytes >= 0x80 included) passes through verbatim.
void json_escape(std::string &out, std::string const &s)
{
    for (unsigned char c : s)
    {
        switch (c)
        {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20)
            {
                char buf[8];
                std::snprintf(
                    buf, sizeof buf, "\\u%04x", static_cast<unsigned>(c));
                out += buf;
            }
            else
            {
                out += static_cast<char>(c);
            }
            break;
        }
    }
}

void append_json_string(std::string &out, std::string const &s)
{
    out += '"';
    json_escape(out, s);
    out += '"';
}

void append_u64(std::string &out, std::uint64_t v)
{
    out += std::to_string(v);
}

// The duration, exact: ns as fractional microseconds with three
// decimals — dur IS (after − before) in ns, so the ns integer splits
// into whole us and the thousandth remainder with NO rounding at all.
// Locale-independent by construction.
void append_dur_ns(std::string &out, std::uint64_t dur_ns)
{
    char buf[32];
    std::snprintf(buf,
        sizeof buf,
        "%llu.%03llu",
        static_cast<unsigned long long>(dur_ns / 1000),
        static_cast<unsigned long long>(dur_ns % 1000));
    out += buf;
}

} // namespace

// ---- the measured swimlane frames  ----

std::string render_measured_swimlanes(
    MeasuredSwimlanes const &lanes, NodeInventory const &inv)
{
    // Lanes, never vendors or runtimes: every name comes
    // from lane_name; an empty lane renders "(idle)" — the finding the
    // full frame exists to show.
    std::string out;
    for (MeasuredLaneSegments const &row : lanes.lanes)
    {
        out += inv.lane_name(row.lane);
        if (row.segments.empty())
        {
            out += "  (idle)";
        }
        else
        {
            for (MeasuredTraceRecord const &seg : row.segments)
            {
                out += "  [plan" + std::to_string(seg.plan_seq) + " q" +
                    std::to_string(seg.op_position) + " " +
                    kind_name(seg.kind) + " " +
                    std::to_string(seg.start_ns) + ".." +
                    std::to_string(seg.end_ns) + "]";
            }
        }
        out += "\n";
    }
    return out;
}

// ---- the join against the simulated trace ----

TraceJoin join_measured_and_simulated(
    std::vector<MeasuredTraceRecord> const &measured,
    std::vector<SimPlanTrace> const &simulated)
{
    // Keyed by the op identity — (plan_seq, queue position), the join
    // surface the two artifacts share — with either side's record
    // beside the other. std::map: the rows come out in ascending
    // identity order, deterministically.
    struct Side
    {
        MeasuredTraceRecord const *m = nullptr;
        SimTraceRecord const *s = nullptr;
    };
    std::map<std::pair<std::uint64_t, std::size_t>, Side> keyed;
    for (MeasuredTraceRecord const &rec : measured)
    {
        keyed[{rec.plan_seq, rec.op_position}].m = &rec;
    }
    for (SimPlanTrace const &plan : simulated)
    {
        for (SimTraceRecord const &rec : plan.records)
        {
            keyed[{plan.plan_seq, rec.op_position}].s = &rec;
        }
    }

    TraceJoin out;
    for (auto const &entry : keyed)
    {
        std::uint64_t const plan_seq = entry.first.first;
        std::size_t const position = entry.first.second;
        Side const &side = entry.second;
        std::string const where = "plan " + std::to_string(plan_seq) +
            " position " + std::to_string(position);

        TraceJoinRow row;
        row.plan_seq = plan_seq;
        row.op_position = position;
        row.measured_present = side.m != nullptr;
        row.simulated_present = side.s != nullptr;

        if (side.m != nullptr)
        {
            row.kind = side.m->kind;
            row.measured_lane = side.m->lane;
            row.measured_start_ns = side.m->start_ns;
            row.measured_end_ns = side.m->end_ns;
        }
        if (side.s != nullptr)
        {
            if (side.m == nullptr)
            {
                row.kind = side.s->kind;
            }
            row.simulated_lane = side.s->lane;
            row.simulated_start_ns = side.s->start_ns;
            row.simulated_end_ns = side.s->end_ns;
        }

        if (side.m != nullptr && side.s != nullptr)
        {
            // The assertion: lane-for-lane equality over the
            // same (plan_seq, position). The times are FACTS that may
            // differ — measured stamps against modeled arithmetic —
            // exposed on the row, never compared.
            row.lane_equal = side.m->lane == side.s->lane;
            if (!row.lane_equal)
            {
                out.mismatches.push_back(
                    "lane mismatch at " + where + ": measured lane " +
                    std::to_string(side.m->lane) + ", simulated lane " +
                    std::to_string(side.s->lane));
            }
            if (side.m->kind != side.s->kind)
            {
                out.mismatches.push_back(
                    "kind mismatch at " + where + ": measured " +
                    kind_name(side.m->kind) + ", simulated " +
                    kind_name(side.s->kind));
            }
        }
        else if (side.m != nullptr)
        {
            out.mismatches.push_back(
                "measured-only record at " + where +
                " — the simulated trace carries no such op");
        }
        else
        {
            out.mismatches.push_back(
                "simulated-only record at " + where +
                " — the measured trace carries no such op");
        }
        out.rows.push_back(row);
    }
    return out;
}

// ---- the tracer: construction and arming ----

PlanTrace::PlanTrace(NodeInventory const &inv, TileTable const &tiles) :
    inv_(&inv),
    tiles_(&tiles)
{
    // The lane set is inventory truth, static for the run; the touched
    // union starts empty at cold start and only ever grows. The tile
    // table rides beside it — the COPY spans' byte source, read at
    // registration only.
    touched_.assign(inv.lane_count(), false);
}

std::optional<std::string> PlanTrace::knob_path()
{
    char const *value = std::getenv("NNHAUL_TRACE");
    if (value == nullptr)
    {
        return std::nullopt; // unset: tracing fully off (the default).
    }
    std::string const knob(value);
    if (knob.empty())
    {
        std::fprintf(stderr,
            "[nnhaul] NNHAUL_TRACE is empty; plan tracing stays off\n");
        return std::nullopt;
    }
    if (knob == "0")
    {
        return std::nullopt;
    }
    if (knob == "1")
    {
        return std::string("nnhaul-trace.json");
    }
    return knob; // the dump path verbatim.
}

void PlanTrace::arm(std::string path)
{
    armed_ = true;
    path_ = std::move(path);
}

void PlanTrace::disarm()
{
    armed_ = false;
    path_.clear();
    blocks_.clear();
    block_index_.clear();
    records_.clear();
    touched_.assign(touched_.size(), false);
    batches_ = 0;
    planned_ops_ = 0;
    issued_ops_ = 0;
    fired_ops_ = 0;
    reconcile_ok_ = true;
}

bool PlanTrace::armed() const
{
    return armed_;
}

// ---- the registration (engine thread, before issue) ----

void PlanTrace::begin_plan(Plan const &plan)
{
    if (!armed_)
    {
        return; // an untraced run registers nothing — nothing will be
                // placed for it either.
    }
    if (block_index_.find(plan.plan_seq) != block_index_.end())
    {
        throw std::runtime_error("plan trace: plan_seq " +
            std::to_string(plan.plan_seq) +
            " registered twice — plans concatenate in issue order, "
            "each seq once");
    }

    auto block = std::make_unique<PlanBlock>();
    block->plan_seq = plan.plan_seq;
    block->ops.assign(plan.queue.size(), OpReg{});
    for (std::size_t p = 0; p < plan.queue.size(); ++p)
    {
        PlanOp const &op = plan.queue[p];
        LaneId const lane = lane_of_op(op, *inv_);
        if (op.kind == PlanOpKind::Copy && lane == kNoLane)
        {
            // The checker owns copy-no-channel (catchable at check
            // time); the tracer never guesses a lane — loud, the belt
            // behind the check.
            throw std::runtime_error("plan trace: COPY at plan " +
                std::to_string(plan.plan_seq) + " position " +
                std::to_string(p) + " over (node " +
                std::to_string(op.u.copy.src.node) + ", chunk " +
                std::to_string(op.u.copy.src.chunk) + ") -> (node " +
                std::to_string(op.u.copy.dst.node) + ", chunk " +
                std::to_string(op.u.copy.dst.chunk) +
                ") — the inventory carries no such channel");
        }
        if (lane == kNoLane)
        {
            // LINK/UNLINK: host-side binding-table transitions at
            // their issue instant — no lane, no stamps, no completion
            // fact.
            block->ops[p] = OpReg{op.kind, kNoLane, false, 0};
            continue;
        }
        // A COPY's occupied byte span: derived HERE from the linked
        // logical chunk's registered tile size — the whole class for a
        // full chunk, the payload remainder for a tail chunk — the fact
        // the dump stamps on the copy event.
        // The tile table aborts loudly on a tile it does not carry:
        // the checker's unknown-tile row is the first line of
        // defense, this lookup the belt.
        std::uint64_t const bytes =
            op.kind == PlanOpKind::Copy
            ? static_cast<std::uint64_t>(tile_chunk_span(
                  tiles_->nbytes(op.u.copy.tile), op.u.copy.index))
            : 0;
        block->ops[p] = OpReg{op.kind, lane, true, bytes};
    }

    block_index_.emplace(plan.plan_seq, blocks_.size());
    blocks_.push_back(std::move(block));
    batches_ += 1;
    planned_ops_ += static_cast<std::uint64_t>(plan.queue.size());
}

void PlanTrace::note_issued(std::size_t issued_ops, bool reconcile_ok)
{
    if (!armed_)
    {
        return;
    }
    issued_ops_ += static_cast<std::uint64_t>(issued_ops);
    reconcile_ok_ = reconcile_ok_ && reconcile_ok;
}

// ---- the harvest (engine thread, at the queue-close join) ----

void PlanTrace::harvest(
    std::uint64_t plan_seq, std::vector<StampFact> const &stamps)
{
    if (!armed_)
    {
        return; // nothing was registered or placed — nothing to read.
    }
    auto it = block_index_.find(plan_seq);
    if (it == block_index_.end())
    {
        throw std::runtime_error("plan trace: harvest of plan_seq " +
            std::to_string(plan_seq) +
            " — no such registered plan (a caller defect)");
    }
    PlanBlock &block = *blocks_[it->second];
    if (block.harvested)
    {
        throw std::runtime_error("plan trace: plan_seq " +
            std::to_string(plan_seq) +
            " harvested twice — the join happens once per plan");
    }
    if (stamps.size() < block.ops.size())
    {
        // The executor's walk covers the whole queue; a short fact
        // vector is a caller defect, loud before anything composes.
        throw std::runtime_error("plan trace: harvest of plan_seq " +
            std::to_string(plan_seq) + " carries " +
            std::to_string(stamps.size()) + " stamp facts for " +
            std::to_string(block.ops.size()) +
            " registered positions — a caller defect");
    }

    // The plan's measured records, composed from the stamp pairs the
    // executor read off the lanes — facts delivered after the
    // barriers that retired every recorded position. A missing pair
    // at an instrumented position is the stamp-fidelity census; an
    // end before its start never happens on one in-order lane and is
    // loud if it ever does.
    std::size_t delivered = 0;
    for (StampFact const &fact : stamps)
    {
        if (fact.start_ns != 0 && fact.end_ns != 0)
        {
            ++delivered;
        }
    }
    for (std::size_t p = 0; p < block.ops.size(); ++p)
    {
        if (!block.ops[p].instrumented)
        {
            continue;
        }
        StampFact const &fact = stamps[p];
        if (fact.start_ns == 0 || fact.end_ns == 0)
        {
            throw std::runtime_error("stamp fidelity: plan " +
                std::to_string(plan_seq) + " position " +
                std::to_string(p) + " (" + kind_name(block.ops[p].kind) +
                " on " + inv_->lane_name(block.ops[p].lane) +
                ") — a stamp pair incomplete after the queue-close "
                "barriers; " + std::to_string(delivered) + " of " +
                std::to_string(block.ops.size()) + " positions "
                "delivered pairs");
        }
        if (fact.end_ns < fact.start_ns)
        {
            throw std::runtime_error("stamp fidelity: plan " +
                std::to_string(plan_seq) + " position " +
                std::to_string(p) +
                " — the after-stamp precedes the before-stamp on one "
                "in-order lane");
        }
        MeasuredTraceRecord const rec{plan_seq,
            p,
            block.ops[p].kind,
            block.ops[p].lane,
            fact.start_ns,
            fact.end_ns,
            block.ops[p].bytes};
        records_.push_back(rec);
        touched_[block.ops[p].lane] = true;
        fired_ops_ += 1; // the after-stamp's completion, verified by
                         // the pair read at the join.
    }
    block.harvested = true;
}

// ---- the products ----

std::vector<MeasuredTraceRecord> const &PlanTrace::measured_records() const
{
    // Engine-thread discipline: composed at harvest, read between
    // harvests.
    return records_;
}

MeasuredSwimlanes PlanTrace::swimlanes(LaneFrame frame) const
{
    MeasuredSwimlanes out;
    out.frame = frame;
    for (std::size_t l = 0; l < touched_.size(); ++l)
    {
        if (frame == LaneFrame::TouchedSoFar && !touched_[l])
        {
            // The concession: the monotone touched-so-far
            // union since cold start — never a per-plan subset, so a
            // lane keeps its row once touched, whatever later plans do.
            continue;
        }
        MeasuredLaneSegments row;
        row.lane = static_cast<LaneId>(l);
        for (MeasuredTraceRecord const &rec : records_)
        {
            if (rec.lane == row.lane)
            {
                // Harvest order is (plan, position) order — lane order
                // for an in-order lane.
                row.segments.push_back(rec);
            }
        }
        out.lanes.push_back(std::move(row));
    }
    return out;
}

std::string PlanTrace::render_swimlanes(LaneFrame frame) const
{
    return render_measured_swimlanes(swimlanes(frame), *inv_);
}

TraceJoin PlanTrace::join_with(
    std::vector<SimPlanTrace> const &simulated) const
{
    return join_measured_and_simulated(measured_records(), simulated);
}

// ---- the observables ----

std::uint64_t PlanTrace::batches() const
{
    return batches_;
}

std::uint64_t PlanTrace::planned_ops() const
{
    return planned_ops_;
}

std::uint64_t PlanTrace::issued_ops() const
{
    return issued_ops_;
}

std::uint64_t PlanTrace::fired_ops() const
{
    return fired_ops_;
}

bool PlanTrace::reconcile_ok() const
{
    return reconcile_ok_;
}

// ---- the dump ----

namespace
{

// One timeline item to emit — an X interval derived from one measured
// record. ts is filled once the origin is known; the sort and the
// format happen only here, at dump.
struct DumpItem
{
    MeasuredTraceRecord rec;
    std::uint64_t ts = 0; // whole us against the dump's origin.
};

} // namespace

bool PlanTrace::dump()
{
    if (!armed_ || path_.empty())
    {
        return false;
    }
    return dump(path_);
}

bool PlanTrace::dump(std::string const &path)
{
    // Engine-thread discipline: the state is read in place; nothing
    // else runs while the sort and the write proceed.
    if (!armed_ || path.empty())
    {
        return false;
    }
    std::vector<MeasuredTraceRecord> const &records = records_;
    std::uint64_t const batches = batches_;
    std::uint64_t const planned = planned_ops_;
    std::uint64_t const issued = issued_ops_;
    std::uint64_t const fired = fired_ops_;
    bool const reconcile_ok = reconcile_ok_;

    // The origin: the first emitted record's before-stamp — every ts
    // is a whole-us floor against it, so the emitted min ts is exactly
    // 0 and wall_clock_span_us (max ts − min ts over the emitted X
    // events) is exact for the checker, all stamps on the one steady
    // clock (every read is the host clock at the op's issue position,
    // device lanes and host lane alike).
    std::uint64_t origin = 0;
    bool first = true;
    for (MeasuredTraceRecord const &rec : records)
    {
        if (first || rec.start_ns < origin)
        {
            origin = rec.start_ns;
            first = false;
        }
    }

    std::vector<DumpItem> items;
    items.reserve(records.size());
    for (MeasuredTraceRecord const &rec : records)
    {
        DumpItem item;
        item.rec = rec;
        item.ts = (rec.start_ns - origin) / 1000;
        items.push_back(item);
    }

    // Per-lane ts monotonicity in file order is the contract; the
    // ORDERING HAPPENS HERE — one global stable sort by ts gives every
    // lane a non-decreasing run (equal ts keep composition order,
    // which is position order on each lane).
    std::stable_sort(items.begin(),
        items.end(),
        [](DumpItem const &a, DumpItem const &b) { return a.ts < b.ts; });

    std::uint64_t span_us = 0;
    if (!items.empty())
    {
        // Sorted: the first item carries the min ts, the last the max.
        span_us = items.back().ts - items.front().ts;
    }

    std::size_t const lane_count = inv_->lane_count();
    std::size_t const node_count = inv_->node_count();

    // Pretty-printed like the previous writer: humans scan it, the
    // tools parse it. The M events first — the process, then one
    // thread per LANE of the inventory (the full frame: an empty lane
    // is itself a finding), tid = lane + 1, named by lane_name —
    // lanes, never vendors or runtimes.
    std::string out;
    out.reserve(512 + items.size() * 160);
    out += "{\n  \"traceEvents\": [\n";
    out += "    {\"ph\": \"M\", \"name\": \"process_name\", \"pid\": 1, "
           "\"ts\": 0, \"args\": {\"name\": \"nnhaul\"}}";
    for (std::size_t l = 0; l < lane_count; ++l)
    {
        out += ",\n    {\"ph\": \"M\", \"name\": \"thread_name\", \"pid\": "
               "1, \"tid\": ";
        append_u64(out, static_cast<std::uint64_t>(l + 1));
        out += ", \"ts\": 0, \"args\": {\"name\": ";
        append_json_string(out, inv_->lane_name(static_cast<LaneId>(l)));
        out += "}}";
    }
    for (DumpItem const &item : items)
    {
        // The X interval: ts = the before-stamp, dur = after − before,
        // and the lane fields — the lane id beside its
        // inventory name — so the tools can diff this measured trace
        // against the simulated swimlanes lane for lane.
        MeasuredTraceRecord const &rec = item.rec;
        out += ",\n    {\"ph\": \"X\", \"name\": ";
        append_json_string(out,
            std::string(kind_name(rec.kind)) + " " +
                std::to_string(rec.plan_seq) + "." +
                std::to_string(rec.op_position));
        out += ", \"cat\": \"";
        out += (rec.lane < node_count) ? "exec" : "chan";
        out += "\", \"pid\": 1, \"tid\": ";
        append_u64(out, static_cast<std::uint64_t>(rec.lane) + 1);
        out += ", \"ts\": ";
        append_u64(out, item.ts);
        out += ", \"dur\": ";
        append_dur_ns(out, rec.end_ns - rec.start_ns);
        out += ", \"args\": {\"plan_seq\": ";
        append_u64(out, rec.plan_seq);
        out += ", \"position\": ";
        append_u64(out, static_cast<std::uint64_t>(rec.op_position));
        out += ", \"kind\": ";
        append_json_string(out, kind_name(rec.kind));
        if (rec.kind == PlanOpKind::Copy)
        {
            // The copy's byte stamp (a derived fact, not a stamp): a
            // COPY moves exactly the occupied span of its source
            // physical chunk (derived from the tile
            // table at registration — the whole class for a full
            // chunk, the tail chunk's payload remainder and nothing
            // more), so a direction's byte sum reads off the trace.
            out += ", \"bytes\": ";
            append_u64(out, rec.bytes);
        }
        out += ", \"lane\": ";
        append_u64(out, rec.lane);
        out += ", \"lane_name\": ";
        append_json_string(out, inv_->lane_name(rec.lane));
        out += "}}";
    }
    out += "\n  ],\n  \"metadata\": {\n";
    out += "    \"format_version\": 1,\n";
    out += "    \"runtime_version\": ";
    append_json_string(out, kRuntimeVersion);
    out += ",\n    \"lane_count\": ";
    append_u64(out, static_cast<std::uint64_t>(lane_count));
    out += ",\n    \"lanes\": [";
    for (std::size_t l = 0; l < lane_count; ++l)
    {
        if (l != 0)
        {
            out += ", ";
        }
        out += "{\"tid\": ";
        append_u64(out, static_cast<std::uint64_t>(l + 1));
        out += ", \"lane\": ";
        append_u64(out, static_cast<std::uint64_t>(l));
        out += ", \"name\": ";
        append_json_string(out, inv_->lane_name(static_cast<LaneId>(l)));
        out += "}";
    }
    out += "],\n    \"wall_clock_span_us\": ";
    append_u64(out, span_us);
    // The plan section: the planned == issued witness, so a reader can
    // verify it from the trace file alone. fired_ops is the
    // completion-fact count: every instrumented op's after-stamp was
    // observed complete at the queue-close join (the read's own
    // verdict), so it equals the instrumented ops of the harvested
    // plans — the witness that no completion went unread.
    out += ",\n    \"plan\": {\n";
    out += "      \"batches\": ";
    append_u64(out, batches);
    out += ",\n      \"planned_ops\": ";
    append_u64(out, planned);
    out += ",\n      \"issued_ops\": ";
    append_u64(out, issued);
    out += ",\n      \"fired_ops\": ";
    append_u64(out, fired);
    out += ",\n      \"reconcile_ok\": ";
    out += reconcile_ok ? "true" : "false";
    out += "\n    }\n";
    out += "  }\n}\n";

    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (f == nullptr)
    {
        std::fprintf(stderr,
            "[nnhaul] plan trace write failed: cannot open %s (%s)\n",
            path.c_str(),
            std::strerror(errno));
        return false;
    }
    bool ok = std::fwrite(out.data(), 1, out.size(), f) == out.size();
    if (std::fclose(f) != 0)
    {
        ok = false;
    }
    if (!ok)
    {
        std::fprintf(stderr,
            "[nnhaul] plan trace write failed: short write to %s (%s)\n",
            path.c_str(),
            std::strerror(errno));
        return false;
    }
    return true;
}

} // namespace nnhaul
