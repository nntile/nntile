/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * chunk-graph executor's tracer over the LANE SET: the measured trace,
 * beside the simulator of src/timing.hh.
 *
 * @file src/plan_trace.hh
 * The tracer of the plan: the stamp facts, the queue-close harvest, and
 * the swimlane rendering over the lane set. The tracing rules are
 * vendor-neutral: an op's interval is captured by stamps placed IN the
 * op's lane at issue positions; each stamp observes the clock AT its
 * lane position; an interval runs from a stamp to the next stamp on
 * that lane; stamps are FACTS, never inputs — they never gate,
 * skip, or reorder an op. Per-op stamps ride the op's OWN lane, armed
 * with the tracer — every lane's stamp is the host clock read at the
 * op's ISSUE position, the submission into the lane being in the lane's
 * order by construction; the host lane's worker reads its own clock at
 * its FIFO positions. The op's measured interval is the tile it owns of
 * the lane's issue order — [its stamp, the next stamp on that lane],
 * the plan's tail stamp closing the last tile; the per-op COMPLETION
 * fact is the pair read at the queue-close join. The facts are read
 * ONLY there, never before — after the barriers that retire every
 * recorded position; the engine plane reads nothing mid-flight. The
 * swimlane framing renders one lane per lane of the inventory, the full
 * frame the default, the monotone touched-so-far union the one
 * concession, names lanes never vendors — the same rendering rules the
 * simulated side follows, so the two artifacts join on (plan_seq, op
 * position) and carry lane ids: lane-for-lane equality is the
 * assertion, times being facts that may differ. The dump keeps the
 * Chrome Trace Event envelope (traceEvents / metadata) with the lane
 * fields added; the old three-lane constants, the vram/ram counters,
 * and the per-op KERNEL/transfer metadata of the previous engine are
 * gone with their sources.
 *
 * Why the stamps are NEITHER host functions NOR timing events (both
 * per-op CUDA mechanisms were evaluated and rejected): a stream-ordered
 * host function stalls its lane while the driver dispatches and runs
 * it, and three placements per instrumented op are enough to double a
 * loop's wall time; a
 * timing-event record stalls no work but taxes the host submission
 * path, whose cost per record grows with the stream's pending depth,
 * and two records per op visibly slow an iteration. The landed read
 * taxes neither path: one host clock read per instrumented op at its
 * issue position, zero CUDA API.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include "backend.hh" // StampFact, EventHandle — the stamp facts
#include "timing.hh"  // SimTraceRecord, LaneFrame — the simulated side of the join

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nnhaul
{

// ---- the measured trace record ----

// One instrumented op's measured fact: the op's identity — plan_seq
// and queue position, the join key against the simulated trace — its
// kind, its LANE (lane_of_op; LINK and UNLINK never appear: no lane,
// no stamps), and its ISSUE READ — one host clock read at the op's
// lane-issue position. The measured interval IS the TILE the op owns of
// its lane's issue order: [its read, the next read on that lane — the
// next op's, or the plan's tail stamp], so the duration INCLUDES the
// lane's queue wait (as in the engine); the times are facts that never
// gate, skip, or reorder anything, and they all sit on the ONE
// steady-clock domain the dump renders, device lanes and host lane
// alike.
struct MeasuredTraceRecord
{
    std::uint64_t plan_seq = 0;
    std::size_t op_position = 0;
    PlanOpKind kind = PlanOpKind::TileOp;
    LaneId lane = kNoLane;
    std::uint64_t start_ns = 0; // the op's issue read (tile start)
    std::uint64_t end_ns = 0;   // the next read on the lane (tile end)
    std::uint64_t bytes = 0;    // a COPY's occupied span, derived at
                                // registration from the tile table;
                                // 0 for the other kinds
};

// ---- the measured swimlane frames  ----

// One lane's row: the lane and its segments — the measured records on
// it, in harvest order, which is (plan, position) order and therefore
// lane order for an in-order lane.
struct MeasuredLaneSegments
{
    LaneId lane = kNoLane;
    std::vector<MeasuredTraceRecord> segments;
};

// The structured swimlane record: one row per lane of the frame, in
// ascending LaneId order — Full, one row per lane of the inventory,
// the DEFAULT (an empty lane is itself a finding: a node or channel
// the planner left idle); TouchedSoFar, the one concession, the
// monotone union of every lane a measured op has landed on since cold
// start, never a per-plan subset. The text render names lanes via
// NodeInventory::lane_name — never vendors or runtimes.
struct MeasuredSwimlanes
{
    LaneFrame frame = LaneFrame::Full;
    std::vector<MeasuredLaneSegments> lanes;
};

// The text render of a structured record: one line per lane of the
// frame, each segment as [plan<p> q<position> <kind> <start>..<end],
// an empty lane rendered "(idle)" — the finding the full frame exists
// to show.
std::string render_measured_swimlanes(
    MeasuredSwimlanes const &lanes, NodeInventory const &inv);

// ---- the join against the simulated trace ----

// The simulated side of the join, one plan's slice of the run trace:
// SimTraceRecords carry no plan_seq (plans concatenate in issue order,
// src/timing.hh), so the join takes the slices keyed by their plans'
// sequence numbers — records in queue order, positions dense.
struct SimPlanTrace
{
    std::uint64_t plan_seq = 0;
    std::vector<SimTraceRecord> records;
};

// One joined op: both sides' facts beside each other. The join asserts
// lane-for-lane equality — lane_equal, over the same (plan_seq,
// position); the times are FACTS that may differ (measured stamps
// against modeled arithmetic), so both pairs are exposed and never
// compared. A row with one side absent carries that side's lanes as
// kNoLane and its times as 0.
struct TraceJoinRow
{
    std::uint64_t plan_seq = 0;
    std::size_t op_position = 0;
    bool measured_present = false;
    bool simulated_present = false;
    PlanOpKind kind = PlanOpKind::TileOp;
    LaneId measured_lane = kNoLane;
    LaneId simulated_lane = kNoLane;
    bool lane_equal = false;
    // The facts, both sides, never compared (a duration and a modeled
    // end are different kinds of fact):
    std::uint64_t measured_start_ns = 0;
    std::uint64_t measured_end_ns = 0;
    std::uint64_t simulated_start_ns = 0;
    std::uint64_t simulated_end_ns = 0;
};

// The join product: every (plan_seq, position) of either side, in
// ascending order, plus the mismatch ledger — one line per finding: a
// lane mismatch, a kind mismatch, or a record present on one side only.
// lanes_match() is the verdict: true iff the ledger is empty.
struct TraceJoin
{
    std::vector<TraceJoinRow> rows;
    std::vector<std::string> mismatches;

    bool lanes_match() const { return mismatches.empty(); }
};

// The join itself: compose the measured trace against the simulated
// slices — per (plan_seq, position), lane equal (the assertion) and
// the times as facts (exposed, never compared).
TraceJoin join_measured_and_simulated(
    std::vector<MeasuredTraceRecord> const &measured,
    std::vector<SimPlanTrace> const &simulated);

// ---- the tracer (run-scoped, armed/disarmed by the engine) ----

// The measured side's one object, run-scoped like the inventory and
// the registries: constructed at init over the run's NodeInventory,
// armed from the NNHAUL_TRACE knob, fed the plans at their issue, and
// handed the stamp facts at their queue-close joins. Instrumentation
// is armed WITH the tracer: an untraced run places no
// stamps — no clock is read, nothing is observed — and nothing
// here observes anything mid-flight; the facts arrive at the close,
// after the barriers. Thread discipline: EVERY method runs on the engine
// thread — the stamps themselves are written by the SUBMITTING thread
// at the op's lane-issue position (the backend's realization of the
// stamp: one host clock read — the engine thread direct or
// the bridge's waiter in the same issue order, never a device-side
// timestamp) and the facts travel to the harvest through the
// executor's slot, never through this object; there is no lock and
// no driver-thread entry point.
class PlanTrace
{
  public:
    // The inventory (the lane set and its names) and the run's tile
    // table: the lane set is inventory truth, static for the run, and
    // the tile table carries the payload bytes every COPY's occupied
    // span derives from (via tile_chunk_span — the byte stamp the dump
    // carries on the copy events); both are run-scoped, read-only here.
    // The touched union starts empty at cold start and only ever grows.
    PlanTrace(NodeInventory const &inv, TileTable const &tiles);

    PlanTrace(PlanTrace const &) = delete;
    PlanTrace &operator=(PlanTrace const &) = delete;

    // ---- arming (the NNHAUL_TRACE knob semantics, read once at
    //      init; the engine calls this, nothing here touches api.cc).

    // The knob parsed as today: unset or "0" -> none (off); "1" ->
    // "nnhaul-trace.json"; any other non-empty value -> the dump path
    // verbatim; empty -> a stderr warning and none.
    static std::optional<std::string> knob_path();

    // Arm with the dump path. An init-time act; arming mid-run starts
    // the accumulation empty. Re-arming re-arms (state kept).
    void arm(std::string path);

    // Disarm: tracing off, everything recorded so far dropped. The
    // engine's act after the run's final dump — the queue-close
    // barriers have retired every recorded position by then, so no
    // fact can still be in flight.
    void disarm();

    bool armed() const;

    // ---- the registration (engine thread, before issue) ----

    // Register one plan: derives every op's kind, lane (lane_of_op; a
    // COPY naming a channel the inventory lacks is loud — the checker
    // owns copy-no-channel, and the tracer never guesses a lane), and
    // — for a COPY — its occupied byte span from the tile table
    // (tile_chunk_span(nbytes(tile), index); a tile the table does not
    // carry is loud, the belt behind the checker's unknown-tile row).
    // One row per LANE-CARRYING op; LINK/UNLINK register nothing. Loud
    // on a repeated plan_seq (plans concatenate in issue order). A no-op
    // while disarmed (nothing will be placed for it).
    void begin_plan(Plan const &plan);

    // The plan's issue facts (engine thread, after its queue-close
    // reconciliation): the issued-ops count and the reconciliation's
    // ok flag — the planned == issued witness the dump's metadata
    // carries, so a reader can verify it from the trace file alone.
    void note_issued(std::size_t issued_ops, bool reconcile_ok);

    // ---- the harvest (engine thread, at the queue-close join,
    //      AFTER the barriers — never before) ----

    // Compose THIS plan's measured records from the stamp facts the
    // executor read off the lanes — one entry per queue position,
    // {0, 0} exactly where nothing was placed (LINK/UNLINK, or an
    // untraced run's empty walk). A missing pair at an instrumented
    // position after the barriers is the stamp-fidelity census; an
    // end before its start never happens on one in-order lane and is
    // loud if it ever does. Facts, read only now — the read point IS
    // the queue-close join; the per-op completion fact is the
    // after-stamp's completion, verified by the lane's own pair read
    // (an incomplete pair is a barrier defect, loud here).
    void harvest(std::uint64_t plan_seq,
        std::vector<StampFact> const &stamps);

    // ---- the products (facts, traveling beside the run) ----

    // The measured trace so far: the composed records of every
    // harvested plan, in (plan_seq, position) order — run order.
    std::vector<MeasuredTraceRecord> const &measured_records() const;

    // The structured swimlane record over the frame.
    MeasuredSwimlanes swimlanes(LaneFrame frame) const;

    // One text render, explicit frame, the full inventory the default.
    std::string render_swimlanes(LaneFrame frame = LaneFrame::Full) const;

    // The join against the simulated trace: this tracer's measured
    // records against the simulated slices (the free function below,
    // over the tracer's own facts).
    TraceJoin join_with(std::vector<SimPlanTrace> const &simulated) const;

    // ---- the dump ----

    // Write the Chrome Trace Event JSON to `path`: one X interval per
    // measured record — ts/dur DERIVED FROM THE STAMPS alone, per-lane
    // ts sorted at dump (the records compose in run order; the sort
    // keeps every lane's file order non-decreasing against any future
    // composition order) — the lane thread_name metadata for EVERY
    // lane of the inventory (the full frame; an empty lane is a
    // finding), the lane fields on every event (lane id + lane name),
    // the copy events' byte stamp (the COPY's occupied span, derived
    // from the tile table at registration — a full chunk the whole
    // class, a tail chunk its payload remainder — so a direction's
    // byte sum reads off the trace), and the plan
    // metadata
    // block (the planned == issued witness). Dumps ACCUMULATE —
    // nothing is cleared; a later dump carries everything plus what
    // came since. Engine thread. Never throws: a write failure
    // reports to stderr and returns false.
    bool dump(std::string const &path);

    // The armed path's dump — false when not armed.
    bool dump();

    // ---- the observables (diagnostics; the dump is the product) ----

    std::uint64_t batches() const;     // plans registered (batches).
    std::uint64_t planned_ops() const; // planned ops, all batches.
    std::uint64_t issued_ops() const;  // issued ops, all batches.
    std::uint64_t fired_ops() const;   // completion facts observed at
                                       // the joins, all batches (the
                                       // after-stamp's completion,
                                       // verified by the pair read).
    bool reconcile_ok() const;         // AND over the batches' flags.

  private:
    // One op's registered facts: its kind, its lane, and — a COPY's
    // occupied byte span (derived from the tile table at registration);
    // instrumented is false exactly for LINK/UNLINK (no lane — nothing
    // placed).
    struct OpReg
    {
        PlanOpKind kind = PlanOpKind::Link;
        LaneId lane = kNoLane;
        bool instrumented = false;
        std::uint64_t bytes = 0; // a COPY's occupied span; 0 otherwise
    };

    // One plan's block: the per-position registry, kept by unique_ptr
    // (a plan's registry outlives the issue call whatever later plans
    // append — the harvest names it by plan_seq).
    struct PlanBlock
    {
        std::uint64_t plan_seq = 0;
        std::vector<OpReg> ops; // per queue position
        bool harvested = false;
    };

    NodeInventory const *inv_;
    TileTable const *tiles_; // the COPY span's byte source (borrowed,
                             // the run's own table)

    bool armed_ = false;
    std::string path_;
    std::vector<std::unique_ptr<PlanBlock>> blocks_; // append-only
    std::unordered_map<std::uint64_t, std::size_t> block_index_;
    std::vector<MeasuredTraceRecord> records_; // composed at harvest
    std::vector<bool> touched_;                // the monotone union

    std::uint64_t batches_ = 0;     // plans begun.
    std::uint64_t planned_ops_ = 0; // planned ops, all batches.
    std::uint64_t issued_ops_ = 0;  // issued ops, all batches.
    std::uint64_t fired_ops_ = 0;   // completion facts, all batches.
    bool reconcile_ok_ = true;      // AND over the batches' flags.
};

} // namespace nnhaul
