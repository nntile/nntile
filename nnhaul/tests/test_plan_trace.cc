/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file tests the
 * chunk-graph executor's tracer over the lane set.
 *
 * @file tests/test_plan_trace.cc
 * The measured-side tracer's suite (ctest target plan_trace): the
 * NNHAUL_TRACE knob semantics; the facts-at-the-join flow — the
 * harvest composes the measured records from the stamp pairs the
 * executor read off the lanes (facts are read only at the queue-close
 * join, never before — nothing is observable mid-flight), durations
 * end-minus-start, ordered within a lane, the completion facts
 * accounted; the measured swimlane framing (the full frame renders
 * empty lanes as findings, touched-so-far a
 * monotone union); the join against the simulated trace (lane-for-
 * lane equality over (plan_seq, position), a lane mismatch DETECTED,
 * one-side-only records detected, both sides' times exposed as
 * facts); the loud belts (copy-no-channel, a repeated plan_seq, an
 * unknown or double harvest, a short fact vector, stamp fidelity);
 * and the JSON dump round-trip over a temp file — every X ts/dur
 * derived from the stamps, per-lane ts non-decreasing in file order,
 * the lane fields (id + name) on every event and in the thread_name
 * metadata for EVERY lane of the inventory, dumps accumulating. Pure
 * CPU: the facts are built from the test thread's steady clock, no
 * device is touched.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#include "../src/plan_trace.hh"

// check.hh's shared macro pulls the OLD public header (nnhaul/nnhaul.hh)
// beside the new contract types, and the two disagree on nnhaul::Access
// today - this suite needs no runtime guards, so the CHECK macro stands
// here alone, same shape as check.hh's (and test_timing.cc's).
#include "nnhaul/defs.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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

// ---- the fixtures  ----

// The inventory: g0 = 0, g1 = 1, host = 2, the peer matrix fully
// probed - NINE lanes: exec(0..2) then chan(0,1)=3, chan(0,2)=4,
// chan(1,0)=5, chan(1,2)=6, chan(2,0)=7, chan(2,1)=8.
NodeInventory make_inv9()
{
    return NodeInventory(2, {4, 4, 4}, {1, 1, 1, 1});
}

// The classic reduction: one device plus host - exec(0),
// exec(1)=exec(host), chan(0,1), chan(1,0), four lanes.
NodeInventory make_inv4()
{
    return NodeInventory(1, {4, 4}, {0});
}

// The suite's tile table : the payload bytes the COPY
// spans derive from at registration . The plans
// here COPY tiles 7 and 9, both full-class payloads — the byte stamp
// on their copy events is the whole class; the span-tail block below
// registers its own tail tile into its own table.
TileTable make_tiles()
{
    TileTable tiles;
    tiles.add(7, chunk_class_bytes());
    tiles.add(9, chunk_class_bytes());
    return tiles;
}

// A small plan over the nine-lane inventory: a LINK (no lane - nothing
// placed for it), a COPY g0 -> host on chan(0,2) = lane 4, a TILE_OP on
// exec(1) = lane 1, a TILE_OP on exec(host) = lane 2.
Plan make_nine_lane_plan(std::uint64_t seq)
{
    Plan plan;
    plan.plan_seq = seq;
    plan.queue.push_back(make_link(7, 0, 2, 1));
    plan.queue.push_back(make_copy(7, 0, ChunkLoc{0, 0}, ChunkLoc{2, 0}));
    plan.queue.push_back(make_tile_op(3, 1, {}));
    plan.queue.push_back(make_tile_op(5, 2, {}));
    return plan;
}

// One stamp pair as a lane would deliver it: the clock read
// immediately before the op's device work, the read immediately
// after — a sleep stands for the work, so the clock moves and
// same-lane order is observable (each stamp observes the clock AT its
// lane position).
StampFact stamp_fact(std::chrono::microseconds work)
{
    StampFact fact;
    fact.start_ns =
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
    std::this_thread::sleep_for(work);
    fact.end_ns =
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
    return fact;
}

// The per-position fact vector of one plan's walk: zero pairs exactly
// where nothing was placed (LINK/UNLINK — no lane, no stamps), read
// pairs elsewhere — what the executor hands the harvest at the
// queue-close join. `work` supplies each instrumented op's
// device-work duration, in queue order.
std::vector<StampFact> facts_for(
    Plan const &plan, std::vector<std::chrono::microseconds> const &work)
{
    std::vector<StampFact> facts(plan.queue.size());
    std::size_t w = 0;
    for (std::size_t p = 0; p < plan.queue.size(); ++p)
    {
        if (plan.queue[p].kind != PlanOpKind::Link &&
            plan.queue[p].kind != PlanOpKind::Unlink)
        {
            facts[p] = stamp_fact(work.at(w));
            ++w;
        }
    }
    return facts;
}

// ---- the JSON scan helpers (the old suite's own, kept) ----

bool read_file(std::string const &path, std::string &out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return false;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    out = buffer.str();
    return true;
}

// The unsigned integer right after `key` (e.g. "\"ts\": ") on one line
// of the dump; false when the key is absent or no digits follow.
bool uint_after(
    std::string const &line, std::string const &key, std::uint64_t &value)
{
    std::size_t const at = line.find(key);
    if (at == std::string::npos)
    {
        return false;
    }
    std::size_t i = at + key.size();
    while (i < line.size() && line[i] == ' ')
    {
        ++i;
    }
    if (i >= line.size() || line[i] < '0' || line[i] > '9')
    {
        return false;
    }
    std::uint64_t v = 0;
    while (i < line.size() && line[i] >= '0' && line[i] <= '9')
    {
        v = v * 10 + static_cast<std::uint64_t>(line[i] - '0');
        ++i;
    }
    value = v;
    return true;
}

// The "W.FFF" duration of one X line as ns - false unless exactly
// three decimals follow the dot (the format contract).
bool dur_ns_of(std::string const &line, std::uint64_t &ns)
{
    std::size_t const at = line.find("\"dur\": ");
    if (at == std::string::npos)
    {
        return false;
    }
    std::size_t i = at + 7;
    std::uint64_t whole = 0;
    bool any = false;
    while (i < line.size() && line[i] >= '0' && line[i] <= '9')
    {
        whole = whole * 10 + static_cast<std::uint64_t>(line[i] - '0');
        ++i;
        any = true;
    }
    if (!any || i >= line.size() || line[i] != '.')
    {
        return false;
    }
    ++i;
    std::uint64_t frac = 0;
    int digits = 0;
    while (i < line.size() && line[i] >= '0' && line[i] <= '9')
    {
        frac = frac * 10 + static_cast<std::uint64_t>(line[i] - '0');
        ++i;
        ++digits;
    }
    if (digits != 3)
    {
        return false;
    }
    ns = whole * 1000 + frac;
    return true;
}

// What the dump scan collects: the X events per lane (tid = lane + 1),
// their per-lane ts monotonicity in file order, the lane args, the
// thread_name metadata events, and the ts span.
struct Scan
{
    std::size_t x_total = 0;
    std::map<std::uint64_t, std::size_t> x_per_tid;
    std::size_t thread_names = 0;
    std::map<std::string, std::size_t> lane_names_seen;
    bool per_lane_monotone = true;
    bool durs_parse = true;
    bool lane_args_on_every_x = true;
    std::uint64_t min_ts = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_ts = 0;
    bool any_ts = false;
};

bool scan_dump(std::string const &path, Scan &scan)
{
    std::string text;
    if (!read_file(path, text))
    {
        return false;
    }
    std::map<std::uint64_t, std::uint64_t> last_ts;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line))
    {
        if (line.find("\"ph\": \"M\"") != std::string::npos)
        {
            if (line.find("\"thread_name\"") != std::string::npos)
            {
                scan.thread_names += 1;
                std::uint64_t tid = 0;
                if (uint_after(line, "\"tid\": ", tid))
                {
                    // The lane's name lives in the ARGS, beside the
                    // event's own name field ("thread_name").
                    std::size_t const at =
                        line.find("\"args\": {\"name\": \"");
                    if (at != std::string::npos)
                    {
                        std::size_t const begin = at + 18;
                        std::size_t const end = line.find('"', begin);
                        if (end != std::string::npos)
                        {
                            scan.lane_names_seen[line.substr(
                                begin, end - begin)] += 1;
                        }
                    }
                }
            }
            continue;
        }
        if (line.find("\"ph\": \"X\"") == std::string::npos)
        {
            continue; // an unknown extra event: tolerated.
        }
        scan.x_total += 1;
        if (line.find("\"lane\": ") == std::string::npos ||
            line.find("\"lane_name\": \"") == std::string::npos)
        {
            scan.lane_args_on_every_x = false;
        }
        std::uint64_t tid = 0;
        std::uint64_t ts = 0;
        std::uint64_t ns = 0;
        if (!uint_after(line, "\"tid\": ", tid) ||
            !uint_after(line, "\"ts\": ", ts))
        {
            return false;
        }
        if (!dur_ns_of(line, ns))
        {
            scan.durs_parse = false;
        }
        if (last_ts.find(tid) != last_ts.end() && ts < last_ts[tid])
        {
            scan.per_lane_monotone = false; // the file contract.
        }
        last_ts[tid] = ts;
        scan.x_per_tid[tid] += 1;
        scan.min_ts = std::min(scan.min_ts, ts);
        scan.max_ts = std::max(scan.max_ts, ts);
        scan.any_ts = true;
    }
    return true;
}

bool metadata_u64(
    std::string const &text, std::string const &key, std::uint64_t &value)
{
    return uint_after(text, "\"" + key + "\": ", value);
}

void set_env(char const *name, char const *value) { setenv(name, value, 1); }

void unset_env(char const *name) { unsetenv(name); }

} // namespace

int main()
{
    // ---- the knob semantics (NNHAUL_TRACE, read once at init).

    {
        unset_env("NNHAUL_TRACE");
        CHECK(!PlanTrace::knob_path().has_value()); // unset: off.
        set_env("NNHAUL_TRACE", "0");
        CHECK(!PlanTrace::knob_path().has_value()); // "0": off.
        set_env("NNHAUL_TRACE", "1");
        CHECK(PlanTrace::knob_path().has_value());
        CHECK(*PlanTrace::knob_path() == "nnhaul-trace.json");
        set_env("NNHAUL_TRACE", "/tmp/plan-trace-knob.json");
        CHECK(PlanTrace::knob_path().has_value());
        CHECK(*PlanTrace::knob_path() == "/tmp/plan-trace-knob.json");
        set_env("NNHAUL_TRACE", ""); // empty: warn (stderr), stay off.
        CHECK(!PlanTrace::knob_path().has_value());
        unset_env("NNHAUL_TRACE");
    }

    // ---- the tracer over the nine-lane inventory: registration, the
    //      facts-at-the-join flow, the records.

    std::string const dump1 = "plan_trace_meas_1.json";
    std::string const dump2 = "plan_trace_meas_2.json";
    std::string const dump3 = "plan_trace_meas_3.json"; // the armed path.

    NodeInventory inv9 = make_inv9();
    TileTable tiles9 = make_tiles();
    PlanTrace trace(inv9, tiles9);
    trace.arm(dump3);
    CHECK(trace.armed());
    NodeInventory inv_untraced = make_inv9();
    TileTable tiles_untraced = make_tiles();
    CHECK(!PlanTrace(inv_untraced, tiles_untraced).dump(dump1)); // not
    // armed: no dump.

    Plan const plan1 = make_nine_lane_plan(0);
    trace.begin_plan(plan1);

    // "Issue" the plan as the executor would: facts per instrumented
    // position (the LINK places nothing), each pair bracketing real
    // work (a 1-ms sleep — the clock moves, the duration is
    // end-minus-start).
    std::vector<StampFact> const facts1 = facts_for(plan1,
        {std::chrono::milliseconds(1),
            std::chrono::milliseconds(1),
            std::chrono::milliseconds(1)});

    // Mid-flight: NOTHING is observable — the facts travel to the
    // queue-close join through the executor's slot; the tracer holds
    // no mid-flight state (read only at the queue-close join).
    CHECK(trace.measured_records().empty());
    CHECK(trace.fired_ops() == 0);

    trace.note_issued(plan1.queue.size(), true);

    // The join: the records compose, the completion facts account.
    trace.harvest(0, facts1);
    CHECK(trace.measured_records().size() == 3);
    CHECK(trace.fired_ops() == 3);

    {
        std::vector<MeasuredTraceRecord> const &recs =
            trace.measured_records();
        // The COPY on chan(0,2) = lane 4; the TILE_OPs on exec(1) = 1
        // and exec(host) = 2 .
        CHECK(recs[0].plan_seq == 0 && recs[0].op_position == 1);
        CHECK(recs[0].kind == PlanOpKind::Copy);
        CHECK(recs[0].lane == 4);
        CHECK(recs[1].op_position == 2);
        CHECK(recs[1].kind == PlanOpKind::TileOp);
        CHECK(recs[1].lane == 1);
        CHECK(recs[2].op_position == 3);
        CHECK(recs[2].lane == 2);
        // The tracing rule : a stamp pair brackets the
        // op's lane position - the duration IS end - start, and the
        // two stamps really read the clock (a sleep of 1 ms moved it).
        for (MeasuredTraceRecord const &rec : recs)
        {
            CHECK(rec.end_ns >= rec.start_ns);
            CHECK(rec.end_ns - rec.start_ns >= 1000000u); // >= 1 ms.
            CHECK(rec.start_ns != 0 && rec.end_ns != 0);
        }
        // Ordered within a lane: the later op's before-stamp observes
        // the clock after the earlier op's after-stamp - different
        // lanes here, so the assertion is the one ordering the rule
        // gives: positions ascending in harvest order.
        CHECK(recs[0].op_position < recs[1].op_position);
        CHECK(recs[1].op_position < recs[2].op_position);
    }

    // Same-lane ordering, made literal: two ops on exec(1) in issue
    // order bracket disjoint clock intervals (the in-order lane).
    {
        Plan plan2;
        plan2.plan_seq = 1;
        plan2.queue.push_back(make_tile_op(3, 1, {}));
        plan2.queue.push_back(make_tile_op(3, 1, {}));
        trace.begin_plan(plan2);
        std::vector<StampFact> const facts = facts_for(plan2,
            {std::chrono::milliseconds(1), std::chrono::milliseconds(1)});
        CHECK(facts[0].end_ns <= facts[1].start_ns); // the lane's order,
        // read off the facts the walk produced in issue order.
        trace.note_issued(plan2.queue.size(), true);
        trace.harvest(1, facts);
        std::vector<MeasuredTraceRecord> const &recs =
            trace.measured_records();
        CHECK(recs.size() == 5);
        MeasuredTraceRecord const &a = recs[3]; // plan 1, position 0.
        MeasuredTraceRecord const &b = recs[4]; // plan 1, position 1.
        CHECK(a.lane == 1 && b.lane == 1);
        CHECK(a.start_ns < a.end_ns);
        CHECK(b.start_ns < b.end_ns);
        CHECK(b.start_ns >= a.end_ns); // brackets, in lane order.
    }

    // ---- the measured swimlane framing ----

    {
        MeasuredSwimlanes const full = trace.swimlanes(LaneFrame::Full);
        CHECK(full.lanes.size() == inv9.lane_count()); // 9 rows: the
        // full frame, empty lanes included - a finding, not a gap.
        std::size_t idle = 0;
        std::size_t busy = 0;
        for (MeasuredLaneSegments const &row : full.lanes)
        {
            if (row.segments.empty())
            {
                ++idle;
            }
            else
            {
                ++busy;
            }
        }
        CHECK(busy == 3); // exec(1) with 3 ops, exec(host) and
        // chan(0,2) with 1 each.
        CHECK(idle == full.lanes.size() - busy);
        std::string const text = trace.render_swimlanes(LaneFrame::Full);
        CHECK(text.find("exec(1)") != std::string::npos);
        CHECK(text.find("exec(2)") != std::string::npos);
        CHECK(text.find("chan(0,2)") != std::string::npos);
        CHECK(text.find("(idle)") != std::string::npos);
        // Lanes, never vendors .
        CHECK(text.find("cuda") == std::string::npos);
        CHECK(text.find("stream") == std::string::npos);

        MeasuredSwimlanes const touched =
            trace.swimlanes(LaneFrame::TouchedSoFar);
        CHECK(touched.lanes.size() == 3); // the monotone union {1,2,4}.
        CHECK(touched.lanes[0].lane == 1);
        CHECK(touched.lanes[0].segments.size() == 3);
        CHECK(touched.lanes[1].lane == 2);
        CHECK(touched.lanes[1].segments.size() == 1);
        CHECK(touched.lanes[2].lane == 4);
        CHECK(touched.lanes[2].segments.size() == 1);
    }

    // The union is MONOTONE: a third plan touching a fresh lane grows
    // it and keeps every earlier lane - never a per-plan subset.
    {
        Plan plan3;
        plan3.plan_seq = 2;
        plan3.queue.push_back(
            make_copy(9, 0, ChunkLoc{1, 0}, ChunkLoc{2, 1})); // chan(1,2)
        trace.begin_plan(plan3);
        std::vector<StampFact> const facts = facts_for(plan3,
            {std::chrono::milliseconds(1)});
        trace.note_issued(plan3.queue.size(), true);
        trace.harvest(2, facts);

        MeasuredSwimlanes const touched =
            trace.swimlanes(LaneFrame::TouchedSoFar);
        CHECK(touched.lanes.size() == 4); // {1, 2, 4, 6} - grown.
        CHECK(touched.lanes[0].lane == 1);
        CHECK(touched.lanes[1].lane == 2);
        CHECK(touched.lanes[2].lane == 4);
        CHECK(touched.lanes[3].lane == 6);
    }

    // ---- the dump: round-trip over the accumulated facts.

    CHECK(trace.dump(dump1));
    {
        Scan scan;
        CHECK(scan_dump(dump1, scan));
        CHECK(scan.x_total == 6); // 3 + 2 + 1 instrumented ops.
        CHECK(scan.thread_names == inv9.lane_count()); // EVERY lane.
        CHECK(scan.lane_names_seen.size() == inv9.lane_count());
        CHECK(scan.lane_names_seen.count("exec(1)") == 1);
        CHECK(scan.lane_names_seen.count("exec(2)") == 1);
        CHECK(scan.lane_names_seen.count("chan(0,2)") == 1);
        CHECK(scan.lane_names_seen.count("chan(1,2)") == 1);
        CHECK(scan.per_lane_monotone); // the file contract.
        CHECK(scan.durs_parse);        // exact three-decimal us.
        CHECK(scan.lane_args_on_every_x);
        // tid = lane + 1: exec(1) is tid 2 with 3 events, chan(0,2)
        // is tid 5 with 1.
        CHECK(scan.x_per_tid[2] == 3);
        CHECK(scan.x_per_tid[5] == 1);
        CHECK(scan.x_per_tid[3] == 1); // exec(2), plan 0's host op.

        std::string text;
        CHECK(read_file(dump1, text));
        std::uint64_t planned = 0;
        std::uint64_t issued = 0;
        std::uint64_t fired = 0;
        std::uint64_t batches = 0;
        std::uint64_t span = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t lane_count = 0;
        CHECK(metadata_u64(text, "planned_ops", planned));
        CHECK(metadata_u64(text, "issued_ops", issued));
        CHECK(metadata_u64(text, "fired_ops", fired));
        CHECK(metadata_u64(text, "batches", batches));
        CHECK(metadata_u64(text, "wall_clock_span_us", span));
        CHECK(metadata_u64(text, "lane_count", lane_count));
        CHECK(lane_count == inv9.lane_count());
        CHECK(batches == 3);
        CHECK(planned == 7 && issued == 7); // planned == issued (4+2+1).
        CHECK(fired == 6); // the completion facts, all observed.
        CHECK(text.find("\"reconcile_ok\": true") != std::string::npos);
        CHECK(text.find("\"format_version\": 1") != std::string::npos);
        CHECK(text.find("\"runtime_version\": \"") != std::string::npos);
        // The span the checker pins: last ts - first ts over the X
        // events, exactly.
        CHECK(scan.any_ts);
        CHECK(span == scan.max_ts - scan.min_ts);
        // The lane fields, both spellings, on the events.
        CHECK(text.find("\"lane\": 4") != std::string::npos);
        CHECK(text.find("\"lane_name\": \"chan(0,2)\"") !=
              std::string::npos);
    }

    // Dumps ACCUMULATE: nothing cleared, a later dump carries
    // everything plus what came since; the armed path writes the same
    // state byte-identically.
    {
        Plan plan4;
        plan4.plan_seq = 3;
        plan4.queue.push_back(make_tile_op(3, 0, {})); // exec(0).
        trace.begin_plan(plan4);
        std::vector<StampFact> const facts = facts_for(plan4,
            {std::chrono::milliseconds(1)});
        trace.note_issued(plan4.queue.size(), false); // a failed
        // reconciliation ANDs in - the witness is honest.
        trace.harvest(3, facts);
        CHECK(trace.dump(dump2));
        CHECK(trace.dump()); // the armed path.
        Scan scan;
        CHECK(scan_dump(dump2, scan));
        CHECK(scan.x_total == 7);
        CHECK(scan.x_per_tid[1] == 1); // exec(0), tid 1.
        std::string a;
        std::string b;
        CHECK(read_file(dump2, a));
        CHECK(read_file(dump3, b));
        CHECK(a == b); // the armed path wrote the same state.
        std::string text;
        CHECK(read_file(dump2, text));
        CHECK(text.find("\"reconcile_ok\": false") != std::string::npos);
    }

    // ---- the loud belts (never serviced, census-style text).

    {
        NodeInventory inv = make_inv9();
        TileTable tiles_belts = make_tiles();
        PlanTrace tt(inv, tiles_belts);
        tt.arm("plan_trace_belts_unused.json");

        // A COPY naming a channel the inventory lacks: the checker
        // owns copy-no-channel; the tracer never guesses a lane. A
        // same-node COPY names no channel at all .
        Plan broken;
        broken.plan_seq = 0;
        broken.queue.push_back(
            make_copy(7, 0, ChunkLoc{0, 0}, ChunkLoc{0, 1}));
        CHECK(throws_loud(
            [&] { tt.begin_plan(broken); }, "no such channel"));

        // A COPY naming a tile the table does not carry: the checker
        // owns unknown-tile; the tracer's span lookup is the belt.
        Plan unknown_tile;
        unknown_tile.plan_seq = 4;
        unknown_tile.queue.push_back(
            make_copy(77, 0, ChunkLoc{0, 0}, ChunkLoc{2, 0}));
        CHECK(throws_loud([&] { tt.begin_plan(unknown_tile); },
            "unknown-tile"));

        // A repeated plan_seq: plans concatenate, each seq once.
        Plan once = make_nine_lane_plan(0);
        tt.begin_plan(once);
        CHECK(throws_loud([&] { tt.begin_plan(once); }, "twice"));

        // An unknown plan's harvest; a double harvest.
        CHECK(throws_loud([&] { tt.harvest(9, {}); }, "no such registered"));
        std::vector<StampFact> const facts = facts_for(once,
            {std::chrono::milliseconds(1),
                std::chrono::milliseconds(1),
                std::chrono::milliseconds(1)});
        tt.note_issued(once.queue.size(), true);
        tt.harvest(0, facts);
        CHECK(throws_loud([&] { tt.harvest(0, facts); }, "twice"));

        // A fact vector shorter than the queue: a caller defect,
        // loud before anything composes.
        Plan plan5;
        plan5.plan_seq = 2;
        plan5.queue.push_back(make_tile_op(3, 0, {}));
        tt.begin_plan(plan5);
        CHECK(throws_loud([&] { tt.harvest(2, {}); }, "stamp facts"));

        // Stamp fidelity: a plan whose after-stamp never landed — the
        // pair is incomplete after the (simulated) barriers.
        Plan starving;
        starving.plan_seq = 3;
        starving.queue.push_back(make_tile_op(3, 0, {}));
        tt.begin_plan(starving);
        std::vector<StampFact> half_empty(1); // {0, 0}: the missing
        // after-stamp — what the barriers failed to retire.
        tt.note_issued(starving.queue.size(), true);
        CHECK(throws_loud(
            [&] { tt.harvest(3, half_empty); }, "stamp fidelity"));

        std::remove("plan_trace_belts_unused.json");
    }

    // ---- disarm: nothing further records, nothing further dumps.

    {
        CHECK(trace.fired_ops() == 7); // 3 + 2 + 1 + 1 across the plans.
        trace.disarm();
        CHECK(!trace.armed());
        CHECK(trace.measured_records().empty()); // everything dropped.
        CHECK(trace.fired_ops() == 0);
        CHECK(!trace.dump(dump1));
        // A harvest after the disarm: the no-op (nothing was
        // registered or placed — nothing to read).
        std::vector<StampFact> const none;
        trace.harvest(3, none); // returns quietly, records nothing.
        CHECK(trace.measured_records().empty());
        trace.disarm();
    }

    // ---- the join against the simulated trace ----

    {
        // Hand-built sides: the measured facts (stamps) and the
        // simulated products (modeled arithmetic) join on (plan_seq,
        // position); lanes agree; times differ and are NEVER compared.
        std::vector<MeasuredTraceRecord> const measured = {
            {0, 0, PlanOpKind::Copy, 4, 100, 200},
            {0, 2, PlanOpKind::TileOp, 1, 300, 400},
            {1, 0, PlanOpKind::TileOp, 2, 500, 600},
        };
        std::vector<SimPlanTrace> const simulated = {
            {0,
                {{0, PlanOpKind::Copy, 4, 0, 1024},
                    {2, PlanOpKind::TileOp, 1, 1024, 1064}}},
            {1, {{0, PlanOpKind::TileOp, 2, 1064, 1084}}},
        };

        TraceJoin const join =
            join_measured_and_simulated(measured, simulated);
        CHECK(join.lanes_match()); // the check's verdict.
        CHECK(join.rows.size() == 3);
        CHECK(join.rows[0].measured_present && join.rows[0].simulated_present);
        CHECK(join.rows[0].lane_equal);
        CHECK(join.rows[0].measured_lane == 4 &&
              join.rows[0].simulated_lane == 4);
        // Both sides' times exposed as facts, never compared.
        CHECK(join.rows[0].measured_start_ns == 100 &&
              join.rows[0].measured_end_ns == 200);
        CHECK(join.rows[0].simulated_start_ns == 0 &&
              join.rows[0].simulated_end_ns == 1024);
        // Rows ascend by (plan_seq, position).
        CHECK(join.rows[0].plan_seq == 0 && join.rows[0].op_position == 0);
        CHECK(join.rows[1].plan_seq == 0 && join.rows[1].op_position == 2);
        CHECK(join.rows[2].plan_seq == 1 && join.rows[2].op_position == 0);

        // A lane mismatch is DETECTED - the one thing the check asserts.
        std::vector<SimPlanTrace> const wrong_lane = {
            {0,
                {{0, PlanOpKind::Copy, 5, 0, 1024}, // chan(1,0), wrong
                    {2, PlanOpKind::TileOp, 1, 1024, 1064}}},
            {1, {{0, PlanOpKind::TileOp, 2, 1064, 1084}}},
        };
        TraceJoin const mismatch =
            join_measured_and_simulated(measured, wrong_lane);
        CHECK(!mismatch.lanes_match());
        CHECK(mismatch.mismatches.size() == 1);
        CHECK(mismatch.mismatches[0].find("lane mismatch") !=
              std::string::npos);
        CHECK(mismatch.mismatches[0].find("plan 0 position 0") !=
              std::string::npos);
        CHECK(!mismatch.rows[0].lane_equal);

        // One-side-only records are findings too: a measured op with
        // no simulated twin, and a simulated op never measured.
        std::vector<MeasuredTraceRecord> const extra_measured = {
            {0, 0, PlanOpKind::Copy, 4, 100, 200},
            {2, 0, PlanOpKind::Copy, 4, 700, 800}, // measured-only.
        };
        std::vector<SimPlanTrace> const extra_simulated = {
            {0,
                {{0, PlanOpKind::Copy, 4, 0, 1024},
                    {1, PlanOpKind::TileOp, 1, 1024, 1064}}}, // sim-only.
        };
        TraceJoin const ragged =
            join_measured_and_simulated(extra_measured, extra_simulated);
        CHECK(!ragged.lanes_match());
        CHECK(ragged.mismatches.size() == 2);
        CHECK(ragged.rows.size() == 3);
        CHECK(ragged.rows[1].simulated_present &&
              !ragged.rows[1].measured_present);
        CHECK(ragged.rows[2].measured_present &&
              !ragged.rows[2].simulated_present);

        // The tracer's own join surface: its measured records against
        // simulated slices - here the empty trace against itself.
        std::vector<SimPlanTrace> const none;
        CHECK(trace.join_with(none).lanes_match());
        CHECK(trace.join_with(none).rows.empty());
    }

    // ---- the classic reduction : one device plus
    //      host - four lanes, host TILE_OPs first-class.

    {
        NodeInventory inv4 = make_inv4();
        CHECK(inv4.lane_count() == 4);
        CHECK(inv4.lane_name(0) == "exec(0)");
        CHECK(inv4.lane_name(1) == "exec(1)");
        CHECK(inv4.lane_name(2) == "chan(0,1)");
        CHECK(inv4.lane_name(3) == "chan(1,0)");

        TileTable tiles4 = make_tiles();
        PlanTrace tt(inv4, tiles4);
        tt.arm("plan_trace_classic.json");
        Plan plan;
        plan.plan_seq = 0;
        plan.queue.push_back(
            make_copy(7, 0, ChunkLoc{1, 0}, ChunkLoc{0, 0})); // host->g0
        plan.queue.push_back(make_tile_op(5, 1, {}));          // host op
        plan.queue.push_back(
            make_copy(7, 0, ChunkLoc{0, 0}, ChunkLoc{1, 1})); // g0->host
        tt.begin_plan(plan);
        std::vector<StampFact> const facts = facts_for(plan,
            {std::chrono::milliseconds(1),
                std::chrono::milliseconds(1),
                std::chrono::milliseconds(1)});
        tt.note_issued(plan.queue.size(), true);
        tt.harvest(0, facts);

        std::vector<MeasuredTraceRecord> const &recs = tt.measured_records();
        CHECK(recs.size() == 3);
        CHECK(recs[0].lane == 3); // chan(1,0)
        CHECK(recs[1].lane == 1); // exec(host)
        CHECK(recs[2].lane == 2); // chan(0,1)

        CHECK(tt.dump("plan_trace_classic.json"));
        Scan scan;
        CHECK(scan_dump("plan_trace_classic.json", scan));
        CHECK(scan.x_total == 3);
        CHECK(scan.thread_names == 4); // every lane of the inventory.
        CHECK(scan.per_lane_monotone);
        std::string text;
        CHECK(read_file("plan_trace_classic.json", text));
        std::uint64_t lane_count = 0;
        CHECK(metadata_u64(text, "lane_count", lane_count));
        CHECK(lane_count == 4);
        // The copy byte stamps: both COPYs move FULL chunks of tile 7
        // — the occupied span IS the class .
        std::uint64_t copy_bytes = 0;
        CHECK(uint_after(text, "\"bytes\": ", copy_bytes));
        CHECK(copy_bytes == chunk_class_bytes());
        std::remove("plan_trace_classic.json");
    }

    // ---- the copy byte stamp, span-exact: a TAIL tile's COPY (one
    //      chunk whose payload is 3/4 of the class) stamps its
    //      occupied span on the event — the true transferred bytes,
    //      never the class constant .
    {
        NodeInventory inv4 = make_inv4();
        TileTable tiles = make_tiles();
        tiles.add(11, 3 * chunk_class_bytes() / 4); // the tail tile
        PlanTrace tt(inv4, tiles);
        tt.arm("plan_trace_span.json");
        Plan plan;
        plan.plan_seq = 0;
        plan.queue.push_back(make_link(11, 0, 1, 0));
        plan.queue.push_back(
            make_copy(11, 0, ChunkLoc{0, 0}, ChunkLoc{1, 0}));
        tt.begin_plan(plan);
        std::vector<StampFact> const facts = facts_for(plan,
            {std::chrono::milliseconds(1)});
        tt.note_issued(plan.queue.size(), true);
        tt.harvest(0, facts);

        std::vector<MeasuredTraceRecord> const &recs =
            tt.measured_records();
        CHECK(recs.size() == 1);
        CHECK(recs[0].kind == PlanOpKind::Copy);
        CHECK(recs[0].bytes == 3 * chunk_class_bytes() / 4);

        CHECK(tt.dump("plan_trace_span.json"));
        std::string text;
        CHECK(read_file("plan_trace_span.json", text));
        std::uint64_t bytes = 0;
        CHECK(uint_after(text, "\"bytes\": ", bytes));
        CHECK(bytes == 3 * chunk_class_bytes() / 4);
        std::remove("plan_trace_span.json");
    }

    // Keep the dumps of a failing run for the post-mortem.
    if (g_checks_failed == 0)
    {
        std::remove(dump1.c_str());
        std::remove(dump2.c_str());
        std::remove(dump3.c_str());
    }

    std::cout << "plan_trace: checks failed = " << g_checks_failed << "\n";
    return g_checks_failed != 0 ? 1 : 0;
}
