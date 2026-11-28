/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * owed derivation and the event mechanism of the chunk-graph contract:
 * the run-scoped dependency table over physical chunks, the one
 * derivation with the domination fold, and the event registry with
 * displacement-only retirement.
 *
 * @file src/dependency.hh
 * The engine-side dependency layer: DependencyState — the two fact
 * families (a USER per (chunk, lane), a PRODUCER per chunk) as dense
 * arrays over the run's slots, moved only forward by the update
 * tables, with the USER fact never reset (free-list reuse depends on
 * it); Derivation — the one dependency pass: a pure walk of a plan's queue
 * computing every op's folded owed set, the mint flags (an event is
 * minted exactly when a later op on another lane owes the op), and the
 * cross-plan debts that mint at this pass as late records;
 * EventRegistry — the run-scoped, op-ref-keyed mint map whose
 * retirement is displacement only. The engine never schedules here:
 * the queue's total order is the whole dependency input, and nothing
 * in this file names an event, a wait, or a lane as a PLAN field —
 * the events below are derived facts, minted and waited by the
 * executor.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include "backend.hh"
#include "plan.hh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nnhaul
{

// ---- the event handle  ----

// The minted ordering marker's name: no payload, no timestamp, no
// readable state. EventHandle is defined in backend.hh — the abstract
// interface that mints and waits handles, included above; this file's
// registry names handles by that type and never restates it.

// ---- DependencyState  ----

// The run-scoped engine-side dependency table, keyed by PHYSICAL chunks
// — (NodeId, ChunkId), the bytes — never logical chunks. Dense over
// the run's slots: slot = seg_base(node) + chunk, seg_base from the
// inventory's per-node pool cardinalities; lanes dense 0..L-1,
// L = lane_count. Two fact families live here, both moving only
// forward: a USER fact — per (chunk, lane), the latest op on that lane
// that accessed the chunk in ANY mode (a writer is also a user) — and
// a PRODUCER fact — per chunk, the latest op that WROTE it. The table
// starts empty at cold start, is never a plan field, and the checker
// rebuilds it by replaying the same rules.
struct DependencyState
{
    // The run's slot space and lane set, fixed at init: the dense fact
    // arrays are sized once — producer_ per slot, user_ per (slot,
    // lane) = slot * L + lane — every slot kNoRef (no fact).
    explicit DependencyState(NodeInventory const &inv);

    // The queries : kNoRef answers "no fact".
    OpRef producer(ChunkLoc c) const;
    OpRef user(ChunkLoc c, LaneId l) const;

    // The update applied after op O (ref) takes effect on lane `lane`:
    // TILE_OP sets USER(each operand, lane) := O and
    // PRODUCER(each written operand — Write or ReadWrite) := O, every
    // operand a user regardless of access; COPY(s, d) sets
    // USER(s, lane) := O, PRODUCER(d) := O, USER(d, lane) := O; UNLINK
    // clears the unlinked chunk's PRODUCER fact (it ends with the
    // binding); LINK touches nothing. The USER fact NEVER resets — not
    // at UNLINK, not at LINK, not at a plan boundary.
    void after(PlanOp const &op, OpRef ref, LaneId lane);

    // The displacement test of the registry's sweep: true iff any
    // PRODUCER or USER fact names `ref`.
    bool names(OpRef ref) const;

    std::vector<OpRef> producer_; // per chunk slot: the latest writer,
                                  // kNoRef = none
    std::vector<OpRef> user_;     // per (slot, lane): slot * L + lane ->
                                  // latest accesser

    // The slot arithmetic's own facts: per node the prefix of the pool
    // cardinalities, and the lane count L.
    std::vector<std::size_t> seg_base_;
    std::vector<std::size_t> cardinality_;
    LaneId lanes_ = kNoLane;

    // Grow node n's slot range to new_cardinality
    // chunks, re-basing the dense fact arrays onto the new seg_base
    // layout (nodes after n shift; every existing fact moves with its
    // slot). Never shrinks; new slots start kNoRef (no fact).
    void grow_node(NodeId n, std::size_t new_cardinality);

  private:
    // Loud programmer-range misuse (the binding table's discipline): a
    // node not of the inventory, a chunk not of its
    // node's pool, a lane not of the run's lane set — never a fold
    // fault, never serviced.
    std::size_t slot(ChunkLoc c) const;
    std::size_t user_index(ChunkLoc c, LaneId l) const;
};

// ---- the one dependency pass  ----

// A cross-plan mint of THIS pass: when the folded owed set of a plan
// names an op of an EARLIER plan, the record is a LATE RECORD enqueued
// into the owed op's lane at this derivation pass, when the debt first
// appears — sound because the plan is derived only after its
// predecessor's last enqueue and the lane is in-order. The executor
// enqueues it; the pass only marks it and names the owner lane.
struct LateRecord
{
    OpRef op;     // the earlier-plan op first owed across the boundary
    LaneId owner; // its lane — the record is enqueued into it
};

// One plan's derivation output: per queue position the folded owed set
// and the mint flag, plus this pass's late records. The owed sets are
// ordered ascending by lane (one surviving ref per other lane at most);
// LINK and UNLINK owe nothing and are never owed.
//
// Lane segments: when the pass runs with lane segments
// (Derivation::run's second argument, the executor's default),
// consecutive queue positions on ONE lane — with laneless LINK/UNLINK
// between them transparent, exactly the steps that submit nothing —
// form ONE SEGMENT, and inside it the intra-segment domination drop
// fires: a member's owed ref is DROPPED iff an earlier member of the
// same segment already carries a kept ref >= it on the same producer
// lane — that member's wait plus the consumer lane's own in-order
// stream imply the dropped edge a fortiori, so the dropped ref never
// walks, never mints, never enters the registry. Every KEPT ref stays
// in its OWN member's row: no wait moves earlier, no member is delayed
// beyond its own need — the dropped edges hold by stream order alone
// (a strict subset of provable no-ops; on one owner lane ref order is
// op order). The queue's order — the plan content — is untouched:
// segments are pure realization grouping. (The stronger TAIL-FOLD form
// — one head wait of the producer's tail event, one tail mint — was
// rejected: it cut the host walk but gated consumer segments on
// producer tails, destroying compute/copy overlap; see dependency.cc.)
struct PlanDerivation
{
    std::vector<std::vector<OpRef>> owed_set; // folded; empty for
                                              // LINK/UNLINK and for
                                              // dominated members
    std::vector<bool> mints;                  // per position: mint
                                              // immediately after the
                                              // op's own work
    std::vector<LateRecord> late_records;     // the cross-plan mints
                                              // of THIS pass
    // The segment census (diagnosis only, never semantics): the
    // segments the pass grouped, the lane-carrying ops they cover, the
    // per-op raw folded refs before the drop, and the refs the drop
    // kept (the difference = the intra-segment dominated edges that
    // never materialize as wait walks).
    std::size_t segments = 0;
    std::size_t segment_ops = 0;
    std::size_t raw_refs = 0;
    std::size_t head_refs = 0;
};

// The run-scoped derivation pass: called per plan in run-queue (issue)
// order, carrying across plans the dependency state and the run-scoped
// OpRef -> LaneId mapping the fold resolves refs through (an op's lane
// is lane_of_op, fixed when the op is derived). The owed procedure is a
// PURE function of the run queue and the empty base case: the executor
// derives and the checker re-derives the same sets mechanically, op for
// op, fold for fold. No scheduling lives here — the pass reads the
// queue's order and derives; it never reorders, defers, or picks.
class Derivation
{
  public:
    explicit Derivation(NodeInventory const &inv);

    // Walks the plan's queue in order: per op the folded owed set
    // (owed(), the domination fold), then the mint flags — an event IS
    // minted on op P iff a LATER op on a lane != lane(P) has P in its
    // folded owed set: post-fold, never speculative, never for a
    // same-lane debt, never two on one op — then the update tables.
    // Refuses a plan_seq out of run-queue order, loud.
    //
    // `lane_segments` (default true) groups consecutive same-lane ops
    // into one segment and applies the intra-segment domination drop
    // (PlanDerivation's note) — the runtime wait elision made structural
    // at derive time, with the enqueued wait set bit-identical. `false`
    // degenerates every segment to the singleton identity: no drop ever
    // fires, byte-identical to the per-op semantics without segments
    // (kept as the reference fallback).
    PlanDerivation run(Plan const &plan, bool lane_segments = true);

    // Grow node n's slot range in the carried dependency
    // state (dense re-base; see DependencyState::grow_node).
    void grow_state(NodeId n, std::size_t new_cardinality)
    {
        dep_.grow_node(n, new_cardinality);
    }


    // The carried dependency state — the sweep's input and the run's
    // trailing facts.
    DependencyState const &state() const { return dep_; }

  private:
    // The not-yet-derived mark of the dense lane rows below: a value
    // no lawful lane and kNoLane's own belt can produce — the old
    // run-scoped map's MISS, made loud at the same place.
    static constexpr LaneId kUnwrittenLane = static_cast<LaneId>(0xfffeu);

    // owed(): the switch over the op's own rows, the fold keeping per
    // lane only the newest owed ref; same-lane debts elide, LINK/UNLINK
    // owe nothing. Writes the folded row, ascending by lane, STRAIGHT
    // into the caller's owed_set slot — no per-op vector is created.
    void owed_into(PlanOp const &op, LaneId l, std::vector<OpRef> &row);

    // The run-scoped lane of a ref — loud on a ref no pass derived.
    LaneId lane_of(OpRef a) const;

    // The run-scoped minted set (never two events on one op): membership
    // and insert over the dense rows below.
    bool minted(OpRef a) const;
    void set_minted(OpRef a);

    NodeInventory const *inv_ = nullptr; // the run's static inventory
    DependencyState dep_;                // carried across plans

    // The derive fold's session scratch, hoisted out of the per-op loop:
    // the fold runs inside the issue window on the engine thread, and
    // the per-op heap churn — a fresh `latest` vector per op plus one
    // unordered_map node (the lane fact) and, per mint, one
    // unordered_set node — was a measurable per-plan cost. OpRef packs
    // (plan_seq, position) and plans derive in run-queue order, so the
    // run-scoped facts become DENSE rows — per plan one row each, per
    // op one array read/write, identical answers (a stored kNoLane is a
    // LINK/UNLINK ref's own lane, exactly what the map held; an
    // out-of-row ref or the unwritten mark is the map's miss), and
    // nothing allocates in steady state: the rows live for the run,
    // regrowing only where a later plan is longer.
    std::vector<OpRef> owed_latest_;            // per lane: the fold's
                                                // newest owed ref,
                                                // reset per op
    std::vector<std::vector<LaneId>> lane_by_seq_;  // ref -> its lane
    std::vector<std::vector<char>> minted_by_seq_;  // 1 iff the op's
                                                    // mint exists
    // The segment head-row scratch (the same session-scratch
    // discipline): reused per segment, never allocated per op.
    std::vector<OpRef> seg_row_;
    bool have_seq_ = false;            // the run-queue order guard
    std::uint64_t last_seq_ = 0;
};

// ---- the event registry  ----

// One run-scoped, engine-side map, keyed by run-scoped op reference:
// op -> its minted event and the minting lane. The registry's lifetime
// is single-threaded with derivation, so a minted, unretired entry is
// stable across the issue of every wait that owes it. Every wait
// resolves through it; a wait naming an absent entry aborts with
// registry-miss — loud, never serviced. Retirement IS displacement
// only: an event retires iff no fact of the current dependency state
// names its op — nothing else retires anything, completion state chief
// among the forbidden inputs — and an entry whose facts stand forever
// stays until the run ends.
struct EventRegistry
{
    struct Entry
    {
        EventHandle handle; // the mint's name: provisioned at init,
                            // recycled at barrier sweeps
        LaneId owner;       // the minting lane — a late record is
                            // enqueued into it
        EventPool *pool = nullptr; // the handle's home provision — a
                                   // HOST-lane op's twin lives in a
                                   // DEVICE pool though its owner is
                                   // the host lane; the sweep releases
                                   // to the home, never the owner's
        std::uint64_t rec_epoch = 0; // the record step's stamp: the
                                     // engine-thread submission order
                                     // of the entry's mint-record
                                     // submit_ call (late record, loop
                                     // mint, or a COPY run's deferred
                                     // member at its flush), 0 until
                                     // submitted. On one owner lane a
                                     // smaller epoch certifies the
                                     // op's bytes are stream-ordered
                                     // before the larger epoch's
                                     // record — the certificate behind
                                     // the transitive wait elision (the
                                     // record order itself can invert:
                                     // a direct mint that extends an
                                     // open COPY run stamps before the
                                     // flush that stamps the run's
                                     // earlier members — see
                                     // plan_executor.hh's elision note).
                                     // Never a retirement input (the
                                     // sweep predicate is untouched).
    };
    std::unordered_map<OpRef, Entry> map;

    // Absent -> the abort registry-miss, with the ref rendered.
    Entry const &resolve(OpRef ref) const;

    // The mint, once: loud on a duplicate — never two events on one op.
    void insert(OpRef ref, Entry entry);

    // Stamp an already-inserted entry's record-submission epoch
    // (see Entry::rec_epoch). The ref was inserted at handle
    // acquisition; an absent entry (the hand-broken-wait test seam)
    // is SILENTLY skipped — returns whether the stamp landed, so the
    // executor's dense fast-path row stays silent too (an absent
    // entry never elides; the consumer's resolve() stays the loud
    // abort).
    bool note_record_epoch(OpRef ref, std::uint64_t epoch);

    // Displacement only: drop every entry whose op no fact of dep
    // names; nothing else retires anything. The DISPLACED entries
    // (exactly the dropped set, in SORTED-REFS order — the same sorted
    // vector the mark pass binary-searches) are MOVED out to
    // `displaced` — the close's handle release walks them without
    // re-finding them or copying the whole registry into a ledger. The
    // release ORDER differs from the old ledger's map-iteration order
    // and is inert: the erase set is equal by construction (the mark
    // pass is the inverted names() predicate over the sorted unique
    // refs) and a pool release only clears bits (CudaEventPool is a
    // lease bitmap — future leases read the free SET, never the release
    // sequence).
    void sweep(DependencyState const &dep,
        std::vector<std::pair<OpRef, Entry>> &displaced);
};

} // namespace nnhaul
