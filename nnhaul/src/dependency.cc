/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * owed derivation's acts: the dense fact arrays' slot arithmetic, the
 * update tables, the domination fold, the mint iff, and the event
 * registry's resolution and displacement sweep.
 *
 * @file src/dependency.cc
 * The acts behind src/dependency.hh: DependencyState's construction
 * from the inventory's per-node cardinalities and its after() — the
 * update tables applied row for row, the USER fact never reset;
 * Derivation::run — the one dependency pass: the owed() walk with the
 * same-lane elision and the domination fold, the mint flags (an event
 * is minted exactly when a later op on another lane owes the op), and
 * the cross-plan late records; EventRegistry's resolve (the
 * registry-miss abort), insert (never two on one op), and sweep.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#include "dependency.hh"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace nnhaul
{

namespace
{

// One op ref rendered (the census and every abort name the op):
// OpRef packs (plan_seq << 32) | queue position, so both halves print.
std::string render(OpRef ref)
{
    return "(plan_seq=" + std::to_string(ref >> 32) + ", position=" +
        std::to_string(ref & 0xffffffffu) + ")";
}

} // namespace

// ---- DependencyState ----

DependencyState::DependencyState(NodeInventory const &inv)
    : seg_base_(inv.node_count()),
      cardinality_(inv.node_count()),
      lanes_(inv.lane_count())
{
    // The run's slot space: slot = seg_base(node) + chunk over the
    // inventory's per-node pool cardinalities, seg_base their prefix
    // sums.
    std::size_t base = 0;
    for (std::size_t n = 0; n < inv.node_count(); ++n)
    {
        cardinality_[n] = inv.pool_cardinality(static_cast<NodeId>(n));
        seg_base_[n] = base;
        base += cardinality_[n];
    }
    // The dense fact arrays, every slot kNoRef — the table starts empty
    // at cold start.
    producer_.assign(base, kNoRef);
    user_.assign(base * static_cast<std::size_t>(lanes_), kNoRef);
}

void DependencyState::grow_node(NodeId n, std::size_t new_cardinality)
{
    // The dense re-base: slot = seg_base(node) + chunk,
    // so growing node n shifts every LATER node's base. The facts are
    // the payload — rebuild the arrays at the new cardinalities with
    // each old slot's facts in their new homes, new slots kNoRef.
    if (static_cast<std::size_t>(n) >= seg_base_.size())
    {
        throw std::runtime_error(
            "dependency state: grow_node of node " +
            std::to_string(n) + " outside the inventory");
    }
    if (new_cardinality < cardinality_[n])
    {
        throw std::runtime_error(
            "dependency state: grow_node would shrink node " +
            std::to_string(n) + "'s slot range — refused");
    }
    if (new_cardinality == cardinality_[n])
    {
        return;
    }
    std::size_t const lanes = static_cast<std::size_t>(lanes_);
    std::vector<std::size_t> const old_base = seg_base_;
    std::vector<std::size_t> const old_card = cardinality_;
    std::size_t const old_slots = producer_.size();
    cardinality_[n] = new_cardinality;
    std::size_t base = 0;
    for (std::size_t m2 = 0; m2 < seg_base_.size(); ++m2)
    {
        seg_base_[m2] = base;
        base += cardinality_[m2];
    }
    std::vector<OpRef> new_producer(base, kNoRef);
    std::vector<OpRef> new_user(base * lanes, kNoRef);
    for (std::size_t node = 0; node < cardinality_.size(); ++node)
    {
        for (std::size_t c = 0; c < old_card[node]; ++c)
        {
            std::size_t const old_slot = old_base[node] + c;
            std::size_t const new_slot = seg_base_[node] + c;
            new_producer[new_slot] = producer_[old_slot];
            for (std::size_t l = 0; l < lanes; ++l)
            {
                new_user[new_slot * lanes + l] =
                    user_[old_slot * lanes + l];
            }
        }
    }
    (void)old_slots;
    producer_ = std::move(new_producer);
    user_ = std::move(new_user);
}


std::size_t DependencyState::slot(ChunkLoc c) const
{
    // Programmer range misuse, loud and never serviced: the node must
    // be of the inventory and the chunk of that node's pool (the
    // checker owns the unknown-id censuses; this is the belt against a
    // corrupt index).
    if (static_cast<std::size_t>(c.node) >= seg_base_.size() ||
        static_cast<std::size_t>(c.chunk) >= cardinality_[c.node])
    {
        throw std::runtime_error("dependency state: chunk (node " +
            std::to_string(static_cast<unsigned>(c.node)) + ", chunk " +
            std::to_string(c.chunk) +
            ") is not a slot of the run's inventory");
    }
    return seg_base_[c.node] + c.chunk;
}

std::size_t DependencyState::user_index(ChunkLoc c, LaneId l) const
{
    if (l >= lanes_)
    {
        throw std::runtime_error("dependency state: lane " +
            std::to_string(l) + " is not one of the run's " +
            std::to_string(lanes_) + " lanes");
    }
    return slot(c) * static_cast<std::size_t>(lanes_) + l;
}

OpRef DependencyState::producer(ChunkLoc c) const
{
    return producer_[slot(c)];
}

OpRef DependencyState::user(ChunkLoc c, LaneId l) const
{
    return user_[user_index(c, l)];
}

void DependencyState::after(PlanOp const &op, OpRef ref, LaneId lane)
{
    // The update tables, applied after op O (ref) takes effect on
    // `lane`. Every row only moves facts forward or clears a PRODUCER
    // at UNLINK — the USER fact never resets, not at UNLINK, not at
    // LINK, not at a plan boundary.
    switch (op.kind)
    {
    case PlanOpKind::TileOp:
        // USER(each operand, lane(O)) := O — every operand is a user in
        // ANY mode, Read included; PRODUCER(each written operand) := O
        // — written is a Write or ReadWrite operand.
        for (Operand const &o : op.operands)
        {
            ChunkLoc const c{o.node, o.chunk};
            user_[user_index(c, lane)] = ref;
            if (o.access == Access::Write || o.access == Access::ReadWrite)
            {
                producer_[slot(c)] = ref;
            }
        }
        break;
    case PlanOpKind::Copy:
    {
        // USER(src, lane) := O; PRODUCER(dst) := O; USER(dst, lane) :=
        // O — the copy becomes the source's latest user and the
        // destination's latest producer and user; the source's
        // PRODUCER fact is untouched.
        ChunkLoc const s = op.u.copy.src;
        ChunkLoc const d = op.u.copy.dst;
        user_[user_index(s, lane)] = ref;
        producer_[slot(d)] = ref;
        user_[user_index(d, lane)] = ref;
        break;
    }
    case PlanOpKind::Unlink:
        // The PRODUCER fact dies with the binding — it is a fact about
        // the logical content and void once the binding is gone —
        // while every USER fact of the chunk lives on.
        producer_[slot(ChunkLoc{op.u.unlink.node, op.u.unlink.chunk})] =
            kNoRef;
        break;
    case PlanOpKind::Link:
        // LINK touches nothing.
        break;
    }
}

bool DependencyState::names(OpRef ref) const
{
    // The sweep's displacement test: the two fact families scanned for
    // the ref, the whole table's semantics in one predicate.
    for (OpRef p : producer_)
    {
        if (p == ref)
        {
            return true;
        }
    }
    for (OpRef u : user_)
    {
        if (u == ref)
        {
            return true;
        }
    }
    return false;
}

// ---- Derivation ----

Derivation::Derivation(NodeInventory const &inv)
    : inv_(&inv),
      dep_(inv),
      owed_latest_(static_cast<std::size_t>(inv.lane_count()), kNoRef)
{
}

LaneId Derivation::lane_of(OpRef a) const
{
    // The run-scoped lane fact, dense: OpRef packs (plan_seq,
    // position), so the ref itself addresses its row. kUnwrittenLane
    // is the not-yet-derived mark — the old run-scoped map's MISS,
    // answered with the same loud throw at the same place — while a
    // stored kNoLane is a laneless op's own fact (LINK/UNLINK's refs
    // carried exactly that value in the map) and passes through to
    // owed_into's belt unchanged.
    std::uint64_t const seq = a >> 32;
    std::uint64_t const pos = a & 0xffffffffu;
    if (seq >= lane_by_seq_.size() || pos >= lane_by_seq_[seq].size() ||
        lane_by_seq_[seq][static_cast<std::size_t>(pos)] == kUnwrittenLane)
    {
        // A fact names an op no derivation pass derived — an engine
        // defect, loud, never serviced: the pass walks every plan in
        // run-queue order, so every live fact's op carries a lane here.
        throw std::runtime_error(
            "dependency derivation: op ref " + render(a) +
            " has no lane — a fact names an op no pass derived");
    }
    return lane_by_seq_[seq][static_cast<std::size_t>(pos)];
}

bool Derivation::minted(OpRef a) const
{
    // Membership in the run-scoped minted set, dense: a ref out of
    // the derived rows answers "not minted" — exactly the old set's
    // miss (the set never threw; neither does this).
    std::uint64_t const seq = a >> 32;
    if (seq >= minted_by_seq_.size())
    {
        return false;
    }
    std::uint64_t const pos = a & 0xffffffffu;
    if (pos >= minted_by_seq_[seq].size())
    {
        return false;
    }
    return minted_by_seq_[seq][static_cast<std::size_t>(pos)] != 0;
}

void Derivation::set_minted(OpRef a)
{
    minted_by_seq_[a >> 32][static_cast<std::size_t>(a & 0xffffffffu)] = 1;
}

void Derivation::owed_into(PlanOp const &op, LaneId l,
    std::vector<OpRef> &row)
{
    // owed(): the session scratch owed_latest_ holds, per lane, the
    // newest owed ref on that lane (reset per op — so no vector is
    // created per op); add() elides same-lane debts (the lane's
    // in-order semantics already carry them) and folds by issue
    // position (the latest per lane survives, its completion implying
    // every earlier op of that lane). OpRef packs (plan_seq, position),
    // so integer > IS the position comparison. The result carries at
    // most one ref per other lane; dominated edges are never named and
    // their events never minted.
    std::fill(owed_latest_.begin(), owed_latest_.end(), kNoRef);
    auto add = [&](OpRef a) {
        if (a == kNoRef)
        {
            return; // no fact: no debt
        }
        LaneId const al = lane_of(a);
        if (al == kNoLane)
        {
            // Belt: after() records facts only for TILE_OP and COPY
            // ops, which always carry a lane — a laneless op in a fact
            // is an engine defect, loud.
            throw std::runtime_error(
                "dependency derivation: op ref " + render(a) +
                " is owed but carries no lane — LINK/UNLINK never "
                "enter facts");
        }
        if (al == l)
        {
            return; // same-lane debts elide
        }
        if (owed_latest_[al] == kNoRef || a > owed_latest_[al])
        {
            owed_latest_[al] = a; // the domination fold
        }
    };
    switch (op.kind)
    {
    case PlanOpKind::TileOp:
        // The per-operand rows: a Read owes the latest
        // PRODUCER of the chunk; a Write owes the latest USER on each
        // other lane; a ReadWrite owes both rows.
        for (Operand const &o : op.operands)
        {
            ChunkLoc const c{o.node, o.chunk};
            if (o.access == Access::Read || o.access == Access::ReadWrite)
            {
                add(dep_.producer(c));
            }
            if (o.access == Access::Write || o.access == Access::ReadWrite)
            {
                for (LaneId m2 = 0; m2 < inv_->lane_count(); ++m2)
                {
                    if (m2 != l)
                    {
                        add(dep_.user(c, m2));
                    }
                }
            }
        }
        break;
    case PlanOpKind::Copy:
        // The COPY row: the latest PRODUCER of the source, and the
        // latest USER of the destination on each lane != lane(O).
        add(dep_.producer(op.u.copy.src));
        for (LaneId m2 = 0; m2 < inv_->lane_count(); ++m2)
        {
            if (m2 != l)
            {
                add(dep_.user(op.u.copy.dst, m2));
            }
        }
        break;
    case PlanOpKind::Link:
    case PlanOpKind::Unlink:
        // No lane, never owed, never minted.
        break;
    }
    // The folded row, ascending by lane — the deterministic order —
    // written STRAIGHT into the plan's owed_set slot (clear keeps the
    // fresh row's semantics; the slot owns the row's life, as before).
    row.clear();
    for (LaneId lane = 0; lane < inv_->lane_count(); ++lane)
    {
        if (owed_latest_[lane] != kNoRef)
        {
            row.push_back(owed_latest_[lane]);
        }
    }
}

PlanDerivation Derivation::run(Plan const &plan, bool lane_segments)
{
    // Plans derive in run-queue order, each pass after the
    // predecessor's LAST op is enqueued. A plan_seq out of order would
    // collide the packed ref space — loud, never serviced.
    if (have_seq_ && plan.plan_seq <= last_seq_)
    {
        throw std::runtime_error("dependency derivation: plan_seq " +
            std::to_string(plan.plan_seq) + " follows plan_seq " +
            std::to_string(last_seq_) +
            " — plans derive in run-queue order");
    }
    have_seq_ = true;
    last_seq_ = plan.plan_seq;

    PlanDerivation out;
    out.owed_set.resize(plan.queue.size());
    out.mints.assign(plan.queue.size(), false);

    // This plan's dense fact rows, sized once up front. The lane row
    // is computed for the WHOLE plan before the walk: the
    // segment structure (below) needs every position's lane upfront,
    // and the answer is lane_of_op — a pure function of the op's own
    // fields — so the prefill writes exactly what the
    // walk's incremental write would have written for every legal
    // input (a fact never names a position at or after the consumer
    // that reads it). The rows live for the run like the map/set did;
    // nothing here allocates per op.
    if (lane_by_seq_.size() <= plan.plan_seq)
    {
        lane_by_seq_.resize(plan.plan_seq + 1);
    }
    std::vector<LaneId> &lane_row = lane_by_seq_[plan.plan_seq];
    lane_row.resize(plan.queue.size());
    for (std::size_t p = 0; p < plan.queue.size(); ++p)
    {
        lane_row[p] = lane_of_op(plan.queue[p], *inv_);
    }
    if (minted_by_seq_.size() <= plan.plan_seq)
    {
        minted_by_seq_.resize(plan.plan_seq + 1);
    }
    minted_by_seq_[plan.plan_seq].assign(plan.queue.size(), 0);

    // THE LANE SEGMENTS: one backward pass marks every position with
    // its segment's TAIL — the last same-lane member of the maximal run
    // of consecutive positions on ONE lane, the laneless LINK/UNLINK
    // (and the no-channel COPY belt) transparent between members
    // exactly as they are to a COPY run (they submit nothing, enter no
    // fact, are owed and mint nothing). A laneless position is its own
    // tail (the value is never read — a laneless op enters no fact).
    // `lane_segments == false` forces the singleton identity:
    // tail[p] == p, no member's row is touched.
    //
    // REJECTED FORM (the first candidate): folding the members' rows to
    // ONE head wait of the producer TAIL event cut the host walk but
    // grew the issue window — waiting a producer segment's LAST op gates
    // the consumer's whole segment, and at this plan shape (one exec
    // lane, per-op interleaved copy dependencies) that destroys
    // compute/transfer overlap worth far more than the host gain. The
    // surviving form below drops ONLY the provably dominated refs,
    // keeping every surviving wait AT ITS OWN MEMBER — zero device-side
    // coarsening by construction.
    std::vector<std::uint32_t> tail(plan.queue.size());
    {
        LaneId cur = kNoLane;
        std::size_t cur_tail = 0;
        for (std::size_t p = plan.queue.size(); p-- > 0;)
        {
            if (!lane_segments || lane_row[p] == kNoLane)
            {
                tail[p] = static_cast<std::uint32_t>(p);
                continue;
            }
            if (lane_row[p] == cur)
            {
                tail[p] = static_cast<std::uint32_t>(cur_tail);
            }
            else
            {
                cur = lane_row[p];
                cur_tail = p;
                tail[p] = static_cast<std::uint32_t>(p);
            }
        }
    }

    // Phase 1 — the walk: the per-op raw fold over the facts (owed(),
    // same-lane elision, the domination fold) and the update tables,
    // both per-op exactly as before (an op never owes itself: the fold
    // reads the facts BEFORE this op takes effect).
    for (std::size_t p = 0; p < plan.queue.size(); ++p)
    {
        PlanOp const &op = plan.queue[p];
        LaneId const lane = lane_row[p];
        owed_into(op, lane, out.owed_set[p]);
        out.raw_refs += out.owed_set[p].size();
        dep_.after(op, make_op_ref(plan.plan_seq, p), lane);
    }

    // Phase 2 — the intra-segment domination drop. Per segment, per
    // member in queue order, a ref on producer lane M is DROPPED from
    // the member's row iff an EARLIER member of the SAME segment
    // already carries a kept ref >= it on M: that member's wait plus
    // the consumer lane's own stream order imply this member's edge a
    // fortiori — the runtime wait elision's stream-order argument
    // applied at derive time and restricted to the segment (the drop's
    // soundness is independent of the runtime epoch certificate: on one
    // owner lane ref order IS op order, so the kept wait implies the
    // dropped ref's op executed; a dropped edge may survive the runtime
    // check in the stamp/ref inversion class, a rare steady-state case
    // — but the edge holds a fortiori regardless). The walk census and
    // the dropped refs' mints, registry inserts, and epoch stamps
    // disappear. A ref that is a NEW max on its lane is KEPT in its own
    // member's row — every surviving wait sits at its own member,
    // nothing moves earlier, no member is delayed beyond its own need
    // (the rejected tail-fold's coarsening, structurally absent). The
    // mint iff then runs over the kept rows in discovery order —
    // positions ascending, refs ascending by lane, the order for
    // singleton segments.
    for (std::size_t p = 0; p < plan.queue.size();)
    {
        if (lane_row[p] == kNoLane)
        {
            ++p; // laneless: the raw row is empty by the fold's rule
            continue;
        }
        std::size_t const t = tail[p];
        std::fill(owed_latest_.begin(), owed_latest_.end(), kNoRef);
        for (std::size_t q = p; q <= t; ++q)
        {
            if (lane_row[q] == kNoLane || out.owed_set[q].empty())
            {
                continue;
            }
            seg_row_.clear();
            for (OpRef a : out.owed_set[q])
            {
                LaneId const al = lane_of(a);
                if (al == lane_row[p])
                {
                    continue; // belt: the per-op fold elided same-lane
                }
                if (owed_latest_[al] != kNoRef && a <= owed_latest_[al])
                {
                    continue; // dominated in-segment: covered a
                              // fortiori by the earlier member's kept
                              // wait — dropped, never walked
                }
                seg_row_.push_back(a);
                owed_latest_[al] = a; // a new max: kept, at member q
            }
            if (seg_row_.size() != out.owed_set[q].size())
            {
                out.owed_set[q].assign(
                    seg_row_.begin(), seg_row_.end());
            }
        }
        for (std::size_t q = p; q <= t; ++q)
        {
            for (OpRef a : out.owed_set[q])
            {
                if (minted(a))
                {
                    continue; // several consumers may wait one event
                }
                set_minted(a);
                if ((a >> 32) == plan.plan_seq)
                {
                    out.mints[static_cast<std::size_t>(
                        a & 0xffffffffu)] = true;
                }
                else
                {
                    out.late_records.push_back(LateRecord{a, lane_of(a)});
                }
            }
        }
        ++out.segments;
        for (std::size_t q = p; q <= t; ++q)
        {
            if (lane_row[q] != kNoLane)
            {
                ++out.segment_ops;
            }
        }
        p = t + 1;
    }
    for (std::size_t p = 0; p < out.owed_set.size(); ++p)
    {
        out.head_refs += out.owed_set[p].size();
    }
    return out;
}

// ---- EventRegistry ----

EventRegistry::Entry const &EventRegistry::resolve(OpRef ref) const
{
    auto it = map.find(ref);
    if (it == map.end())
    {
        // The registry-miss abort: loud, with the ref rendered, never
        // serviced.
        throw std::runtime_error("registry-miss: the event registry "
                                 "carries no minted event for op ref " +
            render(ref));
    }
    return it->second;
}

void EventRegistry::insert(OpRef ref, Entry entry)
{
    // Never two events on one op: a duplicate mint is a
    // registry defect, loud.
    if (!map.emplace(ref, entry).second)
    {
        throw std::runtime_error("event registry: duplicate mint for op "
                                 "ref " +
            render(ref) + " — never two events on one op");
    }
}

bool EventRegistry::note_record_epoch(OpRef ref, std::uint64_t epoch)
{
    // The record-submission stamp (Entry::rec_epoch): the caller
    // stamps at the mint-record submit_ call site, in strictly
    // increasing engine-thread order. Absence is SILENTLY skipped:
    // the stamp only ever ENABLES the transitive wait elision (an
    // unstamped ref is never elided against), while a genuinely
    // absent entry is already aborted loudly downstream by the
    // consumer's resolve() (the registry-miss census) — the
    // hand-broken-wait test seam (post-derive erase) must keep that
    // abort as the FIRST failure. Returns whether the stamp landed so
    // the executor's dense fast-path view stays consistent with it.
    auto it = map.find(ref);
    if (it == map.end())
    {
        return false;
    }
    it->second.rec_epoch = epoch;
    return true;
}

void EventRegistry::sweep(DependencyState const &dep,
    std::vector<std::pair<OpRef, Entry>> &displaced)
{
    // Retirement IS displacement only: an entry leaves iff no fact of
    // the current dependency state names its op. Nothing else retires
    // anything — completion state chief among the forbidden inputs —
    // and an entry whose facts stand forever stays until the run ends.
    //
    // The predicate per entry is EXACTLY names(ref) — erased iff no
    // fact cell holds it — but evaluated INVERTED: ONE pass over the
    // fact cells total, each cell's value tested against the registry's
    // refs, instead of one full-cell scan per registry entry (names()
    // walks producer_ + user_, a large cell count at this engine's
    // inventory; the steady registry holds many live entries, so the
    // per-entry shape dominated a close while the whole loop was
    // host-bound). The erased set is IDENTICAL by construction: an
    // entry survives iff some cell's value equals its ref, whichever
    // side drives the comparison. The per-entry names() stays for the
    // audit and any future caller.
    //
    // The DISPLACED entries are MOVED out to `displaced` via node
    // extraction, in the refs vector's order — SORTED-REFS order, since
    // refs was sorted for the mark pass below — so the close's
    // pool-release ORDER differs from the older ledger's map-iteration
    // order. The order change is INERT: the erase SET is identical by
    // construction (the mark pass IS names() inverted, evaluated
    // cell-side over the sorted unique refs — an entry survives iff some
    // cell's value equals its ref, whichever side drives the comparison),
    // and a pool release only CLEARS BITS (CudaEventPool is a lease
    // bitmap; every future lease depends on the free SET alone, never on
    // the release sequence), so no observable state depends on the
    // order. The whole-registry ledger copy and the per-entry find both
    // disappear. (An open-addressing membership table was tried for the
    // mark pass and was SLOWER at both sizes — the data-dependent probe
    // loop mispredicts where the sorted binary search's uniform descent
    // predicts; the binary search stays.)
    if (map.empty())
    {
        return;
    }
    std::vector<OpRef> refs;
    refs.reserve(map.size());
    for (auto const &kv : map)
    {
        refs.push_back(kv.first);
    }
    std::sort(refs.begin(), refs.end());
    std::vector<unsigned char> named(refs.size(), 0);
    auto touch = [&refs, &named](OpRef v) {
        if (v == kNoRef)
        {
            return; // no fact: names nothing
        }
        auto it = std::lower_bound(refs.begin(), refs.end(), v);
        if (it != refs.end() && *it == v)
        {
            named[static_cast<std::size_t>(it - refs.begin())] = 1;
        }
    };
    for (OpRef v : dep.producer_)
    {
        touch(v);
    }
    for (OpRef v : dep.user_)
    {
        touch(v);
    }
    for (std::size_t i = 0; i < refs.size(); ++i)
    {
        if (named[i] != 0)
        {
            continue;
        }
        // The extract cannot miss: refs came from this very map and
        // nothing mutated it since; the node handle hands the mapped
        // Entry to the caller without rehashing the survivors.
        auto nh = map.extract(refs[i]);
        displaced.emplace_back(nh.key(), std::move(nh.mapped()));
    }
}

} // namespace nnhaul
