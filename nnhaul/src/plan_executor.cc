/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * execute-mode issue loop's acts: the lane table's resolution, the
 * boundary check, the mint acquisitions and the late records, the
 * literal per-op issue with the wait matrix and the deferral mechanism,
 * and the queue close.
 *
 * @file src/plan_executor.cc
 * The acts behind src/plan_executor.hh: execute() runs the issue loop
 * line by line — the applied-equals-entry boundary check over ALL
 * nodes before any op issues; Derivation.run — the one dependency pass —
 * then the mint acquisitions from the lane-family pools including the
 * LATE RECORDS enqueued into the owed ops' lanes at this pass (strictly
 * after the owed op's work by enqueue-before-derive), each registered
 * in the one EventRegistry; the loop over the queue in order — derived
 * waits through the registry on the consumer's own lane before its
 * work (a HOST consumer gates on a host flag the producer lane's
 * notify sets at the mint's enqueue time; a DEVICE consumer of a
 * HOST-family mint enters the deferral mechanism — the file comment's
 * decision; otherwise the handle verbatim), the tracer's ONE
 * issue-position stamp per instrumented op — the tile closes at the
 * next issue read or the plan's tail stamp (no per-op completion
 * notify exists, the completion facts ride the queue-close barriers) —
 * the dispatch (LINK/UNLINK table-only; COPY one occupied-span copy —
 * the byte count from the tile table; TILE_OP one launch), the mint
 * immediately after the work, the SAME fold transitions over the
 * applied table with the rows re-evaluated abort-only, the issue-log
 * append; then close_queue — the mechanism's drain, the planned == issued
 * reconciliation, the barriers over ALL lanes DEVICE LANES FIRST and
 * the host lane LAST (a faulting device lane delivers its fault before
 * the host lane's join can hang on a gate whose notify never fired),
 * the registry sweep by displacement with the retired handles released
 * to their pools, and the tracer harvest. Throughout: an enqueue
 * refusal aborts immediately with a census naming the op — kind,
 * position, OpRef — and dependents never run; the barrier is THE
 * detection point, its fault wrapped with the same census.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#include "plan_executor.hh"

#include <deque>

namespace
{
// The breakdown knob's clock (diagnosis only, never control flow):
// steady nanoseconds, local to this file.
std::uint64_t now_ns_local()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}
// the boundary knob, read once per process (the cached-helper
// pattern: value-parsed through env_flag_enabled, so `=0` disables).
bool boundary_breakdown_on() noexcept
{
    static bool const on =
        nnhaul::env_flag_enabled("NNHAUL_BOUNDARY_BREAKDOWN");
    return on;
}
} // namespace

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <utility>

namespace nnhaul
{

namespace
{
} // namespace


namespace
{

// The op kind's census name.
char const *kind_name(PlanOpKind k)
{
    switch (k)
    {
    case PlanOpKind::Link:
        return "LINK";
    case PlanOpKind::Unlink:
        return "UNLINK";
    case PlanOpKind::Copy:
        return "COPY";
    case PlanOpKind::TileOp:
        return "TILE_OP";
    }
    return "?";
}

// An OpRef rendered (the census names the op: plan_seq and position).
std::string render_ref(OpRef ref)
{
    return "OpRef(plan_seq=" + std::to_string(ref >> 32) +
        ", position=" + std::to_string(ref & 0xffffffffu) + ")";
}

} // namespace

// ---- construction: the lane table resolved once, the mechanism started ----

PlanExecutor::PlanExecutor(NodeInventory const &inv,
    std::vector<ExecLane *> exec_lanes,
    HostWorkerLane *host_lane,
    std::vector<TransferChannel *> channels,
    std::vector<EventPool *> device_pools,
    std::vector<ChunkAddresser> addr,
    std::vector<ChunkContig> contig,
    DeferralBridge *bridge,
    EventPool *host_mint_pool,
    std::function<void(EventHandle)> host_mint_record,
    TileTable const &tiles,
    PlanTrace *tracer) :
    inv_(&inv),
    host_lane_(host_lane),
    bridge_(bridge),
    tracer_(tracer),
    tiles_(&tiles),
    addr_(std::move(addr)),
    contig_(std::move(contig)),
    derivation_(inv),
    registry_()
{
    // The gate-diagnosis flag (NNHAUL_GATE_TRACE): the gate histogram.
    gate_trace_ = std::getenv("NNHAUL_GATE_TRACE") != nullptr;
    host_mint_pool_ = host_mint_pool;
    host_mint_record_ = std::move(host_mint_record);
    if (host_mint_pool_ == nullptr || !host_mint_record_)
    {
        throw std::runtime_error(
            "plan executor: the host-mint twins need a device event "
            "pool and a record seam — a HOST-lane op's mint waited by "
            "a DEVICE lane is a STREAM wait, realized as an event the "
            "host worker records at its FIFO position");
    }

    // Every binding defect is loud at construction, never serviced:
    // the executor drives the whole run on this table.
    std::size_t const n = inv.node_count();
    host_exec_lane_ = inv.lane_of_exec(inv.host_id());

    if (host_lane_ == nullptr)
    {
        throw std::runtime_error(
            "plan executor: a null host lane — the host node's execution "
            "lane is one of the run's lanes ");
    }
    if (exec_lanes.size() != n)
    {
        throw std::runtime_error("plan executor: exec_lanes has " +
            std::to_string(exec_lanes.size()) +
            " entries, but the inventory has k + 1 = " +
            std::to_string(n) + " nodes ");
    }
    if (exec_lanes[inv.host_id()] != static_cast<ExecLane *>(host_lane_))
    {
        throw std::runtime_error(
            "plan executor: exec_lanes[host] is not the host worker lane "
            "— the host entry must be the HostWorkerLane itself");
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        if (exec_lanes[i] == nullptr)
        {
            throw std::runtime_error("plan executor: exec_lanes[" +
                std::to_string(i) + "] is null — every node owns an "
                "execution lane ");
        }
    }
    if (device_pools.size() != inv.device_count())
    {
        throw std::runtime_error("plan executor: device_pools has " +
            std::to_string(device_pools.size()) +
            " entries, but the inventory has k = " +
            std::to_string(inv.device_count()) +
            " device nodes ");
    }
    for (EventPool *p : device_pools)
    {
        if (p == nullptr)
        {
            throw std::runtime_error(
                "plan executor: a null device event pool — provisioned "
                "at init, one per device");
        }
    }

    // The channel count: the inventory's connected ordered pairs, and
    // the channels arrive dense in channel-lane id order (ascending
    // (a, b) of the connected pairs — the numbering the inventory
    // itself derives once).
    std::size_t pairs = 0;
    for (std::size_t a = 0; a < n; ++a)
    {
        for (std::size_t b = 0; b < n; ++b)
        {
            if (a != b && inv.has_channel(static_cast<NodeId>(a),
                                      static_cast<NodeId>(b)))
            {
                ++pairs;
            }
        }
    }
    if (channels.size() != pairs)
    {
        throw std::runtime_error("plan executor: channels has " +
            std::to_string(channels.size()) +
            " entries, but the inventory's channel matrix carries " +
            std::to_string(pairs) +
            " connected ordered pairs (dense in channel-lane id "
            "order)");
    }

    if (contig_.size() != n)
    {
        throw std::runtime_error("plan executor: contig has " +
            std::to_string(contig_.size()) +
            " entries, the inventory has " + std::to_string(n) +
            " nodes — one contiguous-extent surface per node");
    }
    if (addr_.size() != n)
    {
        throw std::runtime_error("plan executor: addr has " +
            std::to_string(addr_.size()) +
            " entries, but the inventory has " + std::to_string(n) +
            " nodes — one chunk-address surface per node");
    }
    for (ChunkAddresser const &a : addr_)
    {
        if (!a)
        {
            throw std::runtime_error(
                "plan executor: a null chunk addresser — addr() is "
                "valid and stable for the run's life");
        }
    }

    // The lane table, indexed by LaneId: exec lanes first (the
    // identity mapping), then the channel lanes.
    lanes_.assign(inv.lane_count(), LaneSlot{});
    // The per-lane COPY-run state, sized with the lane table
    // (only chan lanes ever open one; the exec/host slots stand
    // closed forever).
    copy_runs_.assign(inv.lane_count(), CopyRun{});
    // The break census's arm, read once (the hot loop counts
    // only when armed; the census is diagnosis, never control flow).
    copy_run_diag_on_ = env_flag_enabled("NNHAUL_COPY_RUNS");
    // The executor-path diagnosis arm, read once (diagnosis
    // only, never control flow). The falsifier arm (a witness-only
    // COPY wait-walk skip, NNHAUL_COPY_FALSIFIER) was removed after the
    // diagnostic: it perturbed the data gates deliberately and must
    // never ride a candidate binary.
    copy_diag_on_ = env_flag_enabled("NNHAUL_COPY_BREAKDOWN");
    // The transitive wait-elision's state: the per-(consumer,
    // owner) epoch high-water is sized with the lane table and reset
    // per plan (epochs start at 1; 0 means "never waited this plan").
    // NNHAUL_NO_COPY_ELISION is the A/B seam (the NNHAUL_NO_COPY_FUSION
    // precedent): read once, never diagnosed.
    wait_hw_.assign(
        static_cast<std::size_t>(inv.lane_count()) * inv.lane_count(), 0);
    elision_on_ =
        std::getenv("NNHAUL_NO_COPY_ELISION") == nullptr;
    for (std::size_t node = 0; node < n; ++node)
    {
        LaneId const lane = inv.lane_of_exec(static_cast<NodeId>(node));
        LaneSlot &slot = lanes_[lane];
        slot.lane = exec_lanes[node];
        slot.pool = inv.is_device(static_cast<NodeId>(node))
            ? device_pools[node]
            : &host_lane_->pool();
    }
    {
        std::size_t at = 0;
        for (std::size_t a = 0; a < n; ++a)
        {
            for (std::size_t b = 0; b < n; ++b)
            {
                if (a == b || !inv.has_channel(static_cast<NodeId>(a),
                                             static_cast<NodeId>(b)))
                {
                    continue;
                }
                LaneId const lane =
                    inv.lane_of_chan(static_cast<NodeId>(a),
                        static_cast<NodeId>(b));
                if (channels[at] == nullptr)
                {
                    throw std::runtime_error(
                        "plan executor: channels[" + std::to_string(at) +
                        "] (lane " + std::to_string(lane) + ", " +
                        inv.lane_name(lane) + ") is null");
                }
                LaneSlot &slot = lanes_[lane];
                slot.lane = channels[at];
                slot.chan = channels[at];
                // The family pool of a channel lane: the device whose
                // context the channel's stream lives in — the
                // DESTINATION device when both endpoints are devices,
                // the pair's single device when one endpoint is the
                // host. The mint handle's home must be
                // a pool the minted event resolves through from every
                // consumer (the handle discipline; a backend whose
                // wait resolves handles through a lane-local provision
                // must bind the device lanes' provisions so a
                // cross-lane handle resolves to the minted event —
                // with the CUDA realization, one shared provision per
                // process keeps cross-DEVICE waits resolvable;
                // same-device waits, every wait of a one-device
                // inventory included, are native).
                NodeId const family = inv.is_device(static_cast<NodeId>(b))
                    ? static_cast<NodeId>(b)
                    : static_cast<NodeId>(a);
                slot.pool = inv.is_device(family) ? device_pools[family]
                                                  : &host_lane_->pool();
                ++at;
            }
        }
    }

    // The applied table: N segments, all free — the engine performs
    // the run's initial load through the same transitions at init
    // (applied()).
    std::vector<std::size_t> cardinalities;
    cardinalities.reserve(n);
    for (std::size_t node = 0; node < n; ++node)
    {
        cardinalities.push_back(
            inv.pool_cardinality(static_cast<NodeId>(node)));
    }
    applied_.init(cardinalities);

    // The backend-side deferral bridge: borrowed like the lanes (the
    // engine built it over the host lane and wired it into every
    // device lane's barrier — set_deferral_bridge; a loud close-time
    // belt censuses an unwired bridge).
    if (bridge_ == nullptr)
    {
        throw std::runtime_error(
            "plan executor: a null deferral bridge — a DEVICE lane's "
            "wait of a HOST-family mint is a deferred enqueue, realized "
            "by the backend-side bridge the engine "
            "builds over the host lane and wires into the device lanes' "
            "barriers");
    }
}

// ---- the boundary products ----

BindingTable &PlanExecutor::applied()
{
    return applied_;
}

BindingTable const &PlanExecutor::applied() const
{
    return applied_;
}

void PlanExecutor::apply_growth(NodeId n, std::size_t new_cardinality)
{
    // The executor's own world grows with the pools: the
    // applied table (appended slots free — the sparse boundary compare
    // never sees them) and the carried dependency rows.
    applied_.grow_node(n, new_cardinality);
    derivation_.grow_state(n, new_cardinality);
}

DependencyState const &PlanExecutor::dep_state() const
{
    return derivation_.state();
}

EventRegistry const &PlanExecutor::registry() const
{
    return registry_;
}

EventRegistry &PlanExecutor::registry()
{
    return registry_; // the documented test seam; the engine never edits
}

PlanDerivation const &PlanExecutor::last_derivation() const
{
    return last_derivation_;
}

PlanExecutor::PlanIssue const &PlanExecutor::last_issue() const
{
    // The FULL witness of the most recent plan — issue_log and
    // issued_waits included — lives here alone; the run-order history
    // retains the scalar verdict per plan (see close_queue).
    static PlanIssue const none;
    return have_last_ ? last_full_ : none;
}

std::vector<PlanExecutor::PlanIssue> const &PlanExecutor::issues() const
{
    return issues_;
}

std::uint64_t PlanExecutor::ops_issued() const
{
    return ops_issued_;
}

std::uint64_t PlanExecutor::mints_acquired_total() const
{
    return mints_acquired_;
}

std::uint64_t PlanExecutor::handles_released_total() const
{
    return handles_released_;
}

std::uint64_t PlanExecutor::plans_executed() const
{
    return issues_.size();
}

// ---- the census  ----

std::string PlanExecutor::op_line(Plan const &plan, std::size_t position)
{
    if (position >= plan.queue.size())
    {
        return "none";
    }
    return "position " + std::to_string(position) + ", kind " +
        kind_name(plan.queue[position].kind) + ", " +
        render_ref(make_op_ref(plan.plan_seq, position));
}

void PlanExecutor::abort_census(
    Plan const &plan, std::size_t position, std::string const &finding) const
{
    Census c;
    c.fault = finding;
    if (position != kNoPosition)
    {
        c.fault += " — at op " + op_line(plan, position);
    }
    else
    {
        c.fault += " — at plan_seq " + std::to_string(plan.plan_seq);
    }
    if (position != kNoPosition && position < plan.queue.size())
    {
        c.op = plan.queue[position]; // the offending op verbatim
    }
    // The renders: at execution the applied table IS the live fold;
    // the planner's predicted fold travels with the planner — both
    // census columns render the one live table here, per node with
    // each free list beside it.
    c.fold = applied_.render();
    c.applied = applied_.render();
    c.entry = plan.entry;
    if (position == kNoPosition)
    {
        c.prefix = plan.queue; // a plan-level fault: no op applied
    }
    else
    {
        c.prefix.assign(plan.queue.begin(),
            plan.queue.begin() +
                static_cast<std::ptrdiff_t>(
                    position + 1u <= plan.queue.size() ? position + 1u
                                                       : plan.queue.size()));
    }
    c.issue_prefix = cur_ != nullptr ? cur_->issue_log
                                    : std::vector<PlanOp>{};
    c.inputs = "plan_seq " + std::to_string(plan.plan_seq) + ", queue " +
        std::to_string(plan.queue.size()) + " ops, entry " +
        std::to_string(plan.entry.size()) + " bindings; inventory " +
        std::to_string(inv_->node_count()) + " nodes, " +
        std::to_string(inv_->lane_count()) + " lanes; issue log " +
        std::to_string(
            cur_ != nullptr ? cur_->issue_log.size() : 0) +
        " ops; registry " +
        std::to_string(registry_.map.size()) + " entries";
    // Read the fault BEFORE moving the payload: argument evaluation order
    // is unspecified, and reading a moved-from string leaves what() empty.
    std::string const fault = c.fault;
    throw ExecutorCensusError(fault, std::move(c));
}

// ---- the light row re-evaluation  ----

std::string PlanExecutor::light_rows(PlanOp const &op) const
{
    // Abort-only, free redundancy: a checked plan never fires a row
    // here — the applied table advances by the fold's own transitions
    // and equals the fold at every position — so every
    // finding is the census's issue-abort with the row's fault name
    // verbatim. The rows the APPLIED table and the static inventory
    // can answer only: the tile-table and executability rows are the
    // checker's inputs, never the executor's. Evaluated
    // against the PRE-op applied state — an op's preconditions read
    // the fold of all earlier ops; rows before effects.
    switch (op.kind)
    {
    case PlanOpKind::Link:
    {
        NodeId const node = op.u.link.node;
        ChunkId const chunk = op.u.link.chunk;
        if (!inv_->chunk_valid(node, chunk))
        {
            return "unknown-id";
        }
        if (applied_.linked(node, chunk))
        {
            return "double-link";
        }
        if (applied_.binding_of(
                node, LogicalChunk{op.u.link.tile, op.u.link.index}) !=
            std::nullopt)
        {
            return "double-binding";
        }
        return {};
    }
    case PlanOpKind::Unlink:
    {
        NodeId const node = op.u.unlink.node;
        ChunkId const chunk = op.u.unlink.chunk;
        if (!inv_->chunk_valid(node, chunk))
        {
            return "unknown-id";
        }
        if (!applied_.linked(node, chunk))
        {
            return "unlink-free";
        }
        auto at = applied_.at(node, chunk);
        if (!at.has_value() ||
            !(at->first == LogicalChunk{op.u.unlink.tile, op.u.unlink.index}))
        {
            return "identity-mismatch";
        }
        return {};
    }
    case PlanOpKind::Copy:
    {
        ChunkLoc const src = op.u.copy.src;
        ChunkLoc const dst = op.u.copy.dst;
        LogicalChunk const lc{op.u.copy.tile, op.u.copy.index};
        if (!inv_->chunk_valid(src.node, src.chunk) ||
            !inv_->chunk_valid(dst.node, dst.chunk))
        {
            return "unknown-id";
        }
        auto s = applied_.at(src.node, src.chunk);
        if (!s.has_value() || !(s->first == lc))
        {
            return "copy-identity-mismatch";
        }
        auto d = applied_.at(dst.node, dst.chunk);
        if (!d.has_value() || !(d->first == lc))
        {
            return "copy-identity-mismatch";
        }
        if (src.node == dst.node)
        {
            return "copy-same-node";
        }
        if (!inv_->has_channel(src.node, dst.node))
        {
            return "copy-no-channel";
        }
        if (!applied_.flag_actual(src.node, src.chunk))
        {
            return "copy-from-outdated";
        }
        if (applied_.flag_actual(dst.node, dst.chunk))
        {
            return "copy-to-actual";
        }
        return {};
    }
    case PlanOpKind::TileOp:
    {
        NodeId const node = op.u.tile_op.node;
        if (!inv_->node_valid(node))
        {
            return "unknown-id";
        }
        for (Operand const &o : op.operands)
        {
            if (!inv_->chunk_valid(o.node, o.chunk))
            {
                return "unknown-id";
            }
            if (o.node != node)
            {
                return "operand-node-mismatch";
            }
            if (!applied_.linked(o.node, o.chunk))
            {
                return "access-unlinked";
            }
            if ((o.access == Access::Read ||
                    o.access == Access::ReadWrite) &&
                !applied_.flag_actual(o.node, o.chunk))
            {
                return "read-outdated";
            }
        }
        for (std::size_t i = 0; i < op.operands.size(); ++i)
        {
            for (std::size_t j = i + 1; j < op.operands.size(); ++j)
            {
                if (op.operands[i].node == op.operands[j].node &&
                    op.operands[i].chunk == op.operands[j].chunk)
                {
                    return "duplicate-operand";
                }
            }
        }
        for (std::size_t i = 0; i < op.operands.size(); ++i)
        {
            if (op.operands[i].access == Access::Read)
            {
                continue;
            }
            LogicalChunk const lc{op.operands[i].tile, op.operands[i].index};
            for (std::size_t j = 0; j < op.operands.size(); ++j)
            {
                if (j == i || op.operands[j].access == Access::Read)
                {
                    continue;
                }
                if (op.operands[j].tile == lc.tile &&
                    op.operands[j].index == lc.index)
                {
                    return "duplicate-write";
                }
            }
        }
        return {};
    }
    }
    return {};
}

// ---- the fold's transitions over the applied table ----

void PlanExecutor::apply_transition(PlanOp const &op)
{
    // The SAME calls the checker's fold and the planner's state read
    // make (binding_table.hh's contract): a pure O(1) move per effect,
    // with the flag transitions applied in full.
    switch (op.kind)
    {
    case PlanOpKind::Link:
        applied_.link(op.u.link.node, op.u.link.chunk,
            LogicalChunk{op.u.link.tile, op.u.link.index});
        break;
    case PlanOpKind::Unlink:
        applied_.unlink(op.u.unlink.node, op.u.unlink.chunk);
        break;
    case PlanOpKind::Copy:
        applied_.actualize(op.u.copy.dst.node, op.u.copy.dst.chunk);
        break;
    case PlanOpKind::TileOp:
        // Each written binding ACTUAL, then every OTHER
        // linked replica of its logical chunk OUTDATED, actual ones
        // included; Read operands change no flag.
        for (Operand const &o : op.operands)
        {
            if (o.access == Access::Write || o.access == Access::ReadWrite)
            {
                applied_.actualize(o.node, o.chunk);
                applied_.outdate_replicas(
                    LogicalChunk{o.tile, o.index}, o.node, o.chunk);
            }
        }
        break;
    }
}

// ---- the host-flag companion  ----

void PlanExecutor::set_host_flag_hook(void *arg) noexcept
{
    // The host-flag hook body at a lane position: one flag store
    // through the host lane's own seam (the store takes the lane mutex
    // and notifies the gate's condition variable, so a gated worker
    // wakes, never polls). Placement: the notify is enqueued into the
    // PRODUCER's lane immediately AFTER the mint's record — for a
    // same-plan mint at the producer's issue position (after its device
    // work), for a late record at the successor's derivation pass
    // (after the owed op's already-enqueued work), and for a mint of an
    // earlier pass at the pass that first waits it (after the producer
    // lane's whole enqueued tail — conservative, sound by the in-order
    // lane).
    auto *a = static_cast<HostFlagArg *>(arg);
    a->host->set_flag(a->handle);
}

EventHandle PlanExecutor::acquire_host_flag()
{
    // One host flag per waited mint per plan (several host consumers
    // of one mint gate on the one flag), from the host lane's own
    // provision.
    return host_lane_->pool().acquire();
}

void PlanExecutor::enqueue_host_flag_notify(
    LaneId producer_lane, OpRef producer)
{
    // The deque keeps every earlier argument's address stable while
    // later notifies are in flight; the step's tag is the PRODUCER's
    // ref — a refusal inside the deferred notify censuses the op
    // whose mint it serves, whatever its plan.
    cur_->host_flag_args.push_back(
        HostFlagArg{host_lane_, cur_->host_flags.at(producer)});
    HostFlagArg *arg = &cur_->host_flag_args.back();
    submit_(producer_lane, producer,
        [this, producer_lane, arg] {
            lanes_[producer_lane].lane->notify(
                &PlanExecutor::set_host_flag_hook, arg);
        });
}

// ---- the twins: the mint enqueue with its record twin ----

void PlanExecutor::mint_with_twin_(
    LaneId lane, EventHandle handle, OpRef ref)
{
    if (lane != host_exec_lane_)
    {
        lanes_[lane].lane->mint(handle);
        bridge_->mark_mint_recorded(ref);
        return;
    }
    host_lane_->mint(handle,
        [this, handle, ref](EventHandle) {
            host_mint_record_(handle);
            bridge_->mark_mint_recorded(ref);
        });
}

// ---- the submission path: ONE issue order ----

void PlanExecutor::submit_(LaneId lane, OpRef tag, std::function<void()> fn,
    bool closes_copy_run)
{
    // Every lane step that orders the stream closes an open COPY
    // run of THIS lane first: the run's bytes must precede whatever
    // orders or observes them. The absorbable steps — the member ops'
    // own stream waits (the merged bytes ride behind every member's
    // waits) and the tracer's stamps (a host clock read, zero CUDA
    // API) — pass closes_copy_run=false; everything else keeps the
    // default. The flush itself goes through the bridge directly, so
    // this hook never recurses.
    if (closes_copy_run)
    {
        flush_copy_run_(lane);
    }
    // Direct while THIS lane trails nothing — every backend call records
    // and returns (per lane) — and handed to the backend-side
    // bridge's PER-LANE queue in this lane's own issue order once the
    // lane trails (a gate parks only its lane's tail; every other lane
    // keeps issuing on the engine thread).
    // Within the lane the queue IS the issue order, so every record
    // call precedes its wait call on that lane (the mint-capture rule);
    // cross-lane capture safety rides the bridge's
    // executed-record frontier (issue_wait below).
    bridge_->submit(lane, tag, std::move(fn));
}

void PlanExecutor::flush_copy_run_(LaneId lane)
{
    std::uint64_t const t_fl0 = copy_diag_on_ ? now_ns_local() : 0;
    CopyRun &run = copy_runs_[lane];
    if (!run.open)
    {
        return;
    }
    run.open = false;
    // ONE driver call for the whole run, then the members' deferred
    // mint records in issue order — both through the bridge, so a
    // trailing lane keeps the single issue order exactly as the
    // unmerged submits would have ridden it. The mint-capture rule
    // holds a fortiori: the records' calls trail the merged bytes on
    // the same stream, and every consumer of a member's mint was
    // issued behind the run in queue order (waits name strictly
    // earlier ops) — issue_wait's flush hook guarantees the record's
    // call precedes any resolving wait's call anyway.
    TransferChannel *const chan = lanes_[lane].chan;
    void const *const src = run.src;
    void *const dst = run.dst;
    std::size_t const nbytes = run.nbytes;
    OpRef const tag = run.tag;
    std::uint64_t const t_flprep0 = copy_diag_on_ ? now_ns_local() : 0;
    bridge_->submit(
        lane, tag, [chan, src, dst, nbytes] {
            chan->copy_run(src, dst, nbytes);
        });
    if (copy_diag_on_)
    {
        // The direct path (lane not trailing) ran the merged
        // cudaMemcpyAsync inline; a trailing lane deferred it — the
        // number below is the SUBMIT cost either way, the engine-
        // thread driver call in the common case.
        copy_diag_.flcopy_ns += now_ns_local() - t_flprep0;
        copy_diag_.flprep_ns += t_flprep0 - t_fl0;
        ++copy_diag_.flush_n;
    }
    std::uint64_t const t_mrec0 = copy_diag_on_ ? now_ns_local() : 0;
    for (std::pair<OpRef, EventHandle> const &m : run.mints)
    {
        // the deferred member's epoch stamp, AT THE FLUSH's
        // submit call — the record's true stream position (the run
        // closes before any later record-bearing step on this lane,
        // so members land in issue order here, exactly where the
        // unmerged record stood relative to other records).
        stamp_rec_(m.first, lane);
        bridge_->submit(lane, m.first,
            [this, lane, handle = m.second, ref = m.first] {
                mint_with_twin_(lane, handle, ref);
            });
    }
    if (copy_diag_on_)
    {
        copy_diag_.mrec_ns += now_ns_local() - t_mrec0;
        copy_diag_.mrec_n += run.mints.size();
    }
    copy_run_ops_ += run.ops;
    copy_run_calls_saved_ += run.ops - 1u;
    run.mints.clear();
    run.ops = 0;
    run.nbytes = 0;
}

void PlanExecutor::flush_all_copy_runs_()
{
    for (LaneId lane = 0; lane < inv_->lane_count(); ++lane)
    {
        flush_copy_run_(lane);
    }
}

void PlanExecutor::stamp_rec_(OpRef ref, LaneId owner)
{
    // census: every registry mint-record submit passes here.
    ++r2_rec_submits_;
    // Both epoch views from one counter tick (the dense row for
    // the elision fast path, the registry entry for the full path) —
    // identical by construction. The registry note lands FIRST and
    // gates the dense stamp: an entry ABSENT at stamp time (the
    // hand-broken-wait seam) leaves BOTH views silent, so an absent
    // entry can never elide — the consumer's resolve() stays the
    // loud abort. The dense row grows lazily: rows by seq (plans
    // issue ascending; late records stamp EARLIER seqs' cells),
    // cells by position.
    if (!registry_.note_record_epoch(ref, next_rec_epoch_))
    {
        return;
    }
    std::uint64_t const stamped = next_rec_epoch_++;
    std::uint64_t const seq = ref >> 32;
    std::size_t const pos =
        static_cast<std::size_t>(ref & 0xffffffffu);
    if (seq >= rec_epoch_rows_.size())
    {
        rec_epoch_rows_.resize(static_cast<std::size_t>(seq) + 1);
    }
    RecEpochRow &row = rec_epoch_rows_[static_cast<std::size_t>(seq)];
    if (pos >= row.epoch.size())
    {
        row.epoch.resize(pos + 1, 0);
        row.owner.resize(pos + 1, 0);
    }
    row.epoch[pos] = stamped;
    row.owner[pos] = owner;
}

void PlanExecutor::submit_host_gate_(
    LaneId lane, OpRef tag, EventHandle e, LaneId producer)
{
    // A device-waits-host wait parks ONLY the consumer's own lane
    // : the
    // gate's flag lifts with HOST progress — the engine's host mints
    // are its own no-op acts, microseconds — so the park is bounded,
    // never by device completion. The lane trails from this step.
    bridge_->gate(lane, tag, e, producer);
}

// ---- the wait matrix  ----

void PlanExecutor::issue_wait(
    Plan const &plan, std::size_t position, LaneId consumer, OpRef ref)
{
    // Exactly one wait per folded, cross-lane, owed op, in the
    // consumer's own lane, BEFORE its own device work. The
    // mint-capture guarantee is the executor's alone: every waited
    // record was enqueued strictly earlier — same-
    // plan mints at the producer's earlier queue position, late
    // records and earlier-pass mints at this plan's derivation pass.
    // CUDA never masks a late record: cudaStreamWaitEvent on a
    // never-recorded event does not gate, and the registry-miss abort
    // is the guard, never the driver.
    OpRef const tag = make_op_ref(plan.plan_seq, position);
    // The COPY-consumer split (diagnosis only): the resolution
    // walk (registry + the flush-hook scan) apart from the wait
    // enqueue (the submit incl. the inline cudaStreamWaitEvent).
    // The same centers apply to TILE_OP consumers, where the
    // cross-lane copy edges land here.
    bool const copy_diag_copy = copy_diag_on_ && position < plan.queue.size()
        && plan.queue[position].kind == PlanOpKind::Copy;
    bool const k7_tilo = copy_diag_on_ && position < plan.queue.size()
        && plan.queue[position].kind == PlanOpKind::TileOp;
    std::uint64_t const copy_diag_t0 = (copy_diag_copy || k7_tilo) ? now_ns_local() : 0;
    auto copy_diag_res_done = [&]() {
        if (copy_diag_copy)
        {
            copy_diag_.wres_ns += now_ns_local() - copy_diag_t0;
        }
        if (k7_tilo)
        {
            copy_diag_.tilo_wres_ns += now_ns_local() - copy_diag_t0;
        }
    };
    auto copy_diag_all_done = [&]() {
        if (copy_diag_copy)
        {
            copy_diag_.wait_ns += now_ns_local() - copy_diag_t0;
            ++copy_diag_.wait_n;
        }
        if (k7_tilo)
        {
            copy_diag_.tilo_wait_ns += now_ns_local() - copy_diag_t0;
            ++copy_diag_.tilo_wait_n;
        }
    };
    if (consumer == host_exec_lane_)
    {
        // A HOST-lane consumer: it gates on a HOST flag — never on a
        // device event — the flag set by the producer lane's notify
        // enqueued at the mint's enqueue time (a callback, never a
        // poll). The gate rides the SAME submission path as every other
        // enqueue —
        // pushed directly while the plan trails nothing, FIFO-ordered
        // behind the parked gates once it does — so it can never
        // overtake the deferred notify that sets its flag (the one
        // issue order; a bypass here deadlocks legal mixed plans).
        auto it = cur_->host_flags.find(ref);
        if (it == cur_->host_flags.end())
        {
            // Belt: the derivation pass acquires a flag for every ref
            // a host-lane op owes — an absent flag is an engine
            // defect, loud, never serviced.
            abort_census(plan, position,
                "issue-abort: the host consumer of " + render_ref(ref) +
                    " carries no host flag — the derivation pass acquires "
                    "one per waited mint ");
        }
        EventHandle const flag = it->second;
        // The HOST lane never defers (it parks on its own FIFO's
        // gates, lane-internal): the gate pushes straight into the
        // worker, and the flag-setting notify — enqueued at its
        // EARLIER issue position on the producer's queue — precedes
        // this push on the engine thread, always (the deadlock
        // shape cannot re-form).
        copy_diag_res_done();
        std::uint64_t const copy_diag_tsub = copy_diag_copy ? now_ns_local() : 0;
        submit_(host_exec_lane_, tag, [this, flag] {
            host_lane_->wait(flag);
        });
        if (copy_diag_copy)
        {
            copy_diag_.wsub_ns += now_ns_local() - copy_diag_tsub;
        }
        copy_diag_all_done();
        return;
    }
    // A DEVICE-lane consumer resolves the mint through the registry
    // (absent -> registry-miss, thrown here, wrapped with the op's
    // census by the loop): a HOST-family mint parks the CONSUMER's
    // own lane; a
    // device-family mint waits the handle verbatim — directly when
    // the record already EXECUTED (call-time capture is safe past an
    // executed record), else as the lane's recorded-wait step (the
    // capture-safe discipline: the wait's call cannot precede the
    // record's call).
    //
    // The elision FAST PATH first: a dense-row cell stamped for
    // this ref (epoch != 0) certifies the record's submit_ call
    // happened AND the entry was registered at stamp time — exactly
    // what the registry-epoch check below concludes after the hash
    // resolve — so a covered epoch (<= the pair's high-water) elides
    // with two array reads and no map traffic. A miss (gap seq,
    // never-stamped position, an entry absent at stamp time, or a
    // COPY-run member still deferred) falls through to the full path,
    // byte-identical to the registry-only path. Not covered -> fall
    // through
    // too: the full path enqueues the wait and raises the high-water
    // (the two epoch views are one tick, so it lands on this same
    // value).
    if (elision_on_)
    {
        std::uint64_t const fseq = ref >> 32;
        if (fseq < rec_epoch_rows_.size())
        {
            RecEpochRow const &frow =
                rec_epoch_rows_[static_cast<std::size_t>(fseq)];
            std::size_t const fpos =
                static_cast<std::size_t>(ref & 0xffffffffu);
            if (fpos < frow.epoch.size() && frow.epoch[fpos] != 0)
            {
                std::uint64_t const fe = frow.epoch[fpos];
                std::uint64_t &fhw =
                    wait_hw_[static_cast<std::size_t>(consumer) *
                            inv_->lane_count() +
                        frow.owner[fpos]];
                if (fe <= fhw)
                {
                    ++k9_elided_;
                    ++k9_fast_elided_;
                    copy_diag_res_done();
                    copy_diag_all_done();
                    return;
                }
            }
        }
    }
    EventRegistry::Entry const &entry = registry_.resolve(ref);
    // A wait naming a ref whose mint record may sit DEFERRED in
    // the owner lane's open COPY run: flush that run FIRST, so the
    // record's call precedes this wait's call — the mint-capture rule
    //  keeps exactly the unmerged enqueue order. Only
    // a mint actually deferred in the run needs the flush: any other
    // ref's record was submitted at its own earlier position, already
    // ahead of this wait in the single issue order (the flush would
    // merely break the run — the scan below is the whole cost, a tiny
    // vector).
    // (diagnostic double-count, accepted: when this hook fires — at most
    // once per run, about once per plan — the flush's time is ALSO inside wres_ns
    // below, since copy_diag_res_done() closes over the whole resolution
    // walk including this block.)
    bool copy_diag_hook_fired = false;
    {
        CopyRun &run = copy_runs_[entry.owner];
        if (run.open)
        {
            for (std::pair<OpRef, EventHandle> const &m : run.mints)
            {
                if (m.first == ref)
                {
                    flush_copy_run_(entry.owner);
                    copy_diag_hook_fired = true;
                    break;
                }
            }
        }
    }
    copy_diag_res_done();
    if (copy_diag_copy)
    {
        if (entry.owner == consumer)
        {
            ++copy_diag_.samelane; // the same-lane elision's leak census
        }
        if (copy_diag_hook_fired)
        {
            ++copy_diag_.flushhook;
        }
    }
    // THE TRANSITIVE STREAM-WAIT ELISION: if this
    // consumer lane already enqueued — or parked, in FIFO position —
    // a wait for an event whose record provably sits LATER on the
    // owner's stream than this ref's record (the epoch certificate,
    // plan_executor.hh), this wait is a provable no-op behind it: the
    // consumer's work cannot start before the already-waited event
    // fires, and that firing implies this ref's record executed (one
    // owner lane, in-order stream). The elided edge holds a fortiori;
    // the stream-wait-event-only discipline keeps every surviving
    // edge. Never elided when the record is unstamped (epoch 0: not
    // yet submitted — the producer-event rule) or the A/B seam
    // (NNHAUL_NO_COPY_ELISION) is armed.
    if (elision_on_ && entry.rec_epoch != 0)
    {
        std::uint64_t &hw = wait_hw_[static_cast<std::size_t>(consumer) *
                inv_->lane_count() +
            entry.owner];
        if (entry.rec_epoch <= hw)
        {
            ++k9_elided_;
            copy_diag_all_done();
            return;
        }
        hw = entry.rec_epoch;
    }
    // The elision census: Every wait that ENQUEUES counts here — an epoch-
    // stamped uncovered one (the high-water just raised above), an
    // UNSTAMPED one (epoch 0: the record's submit_ call has not run
    // yet — the producer-event rule keeps it verbatim), or the A/B
    // seam's verbatim shape — so elided + waited = the plan's total
    // derived stream waits.
    ++k9_waited_;
    if (entry.owner == host_exec_lane_)
    {
        // A HOST-lane op's mint, waited by a DEVICE lane: the mint's
        // TWIN is a device event the host worker records at the op's
        // FIFO position — once that record EXECUTED (the mark), the
        // consumer's wait is a PLAIN STREAM WAIT (sync between streams is
        // the non-blocking stream wait for event), direct, no gate, no
        // deferral; the capture rule holds
        // (the wait captures the executed record). The not-yet-
        // recorded window defers as the lane's recorded-wait — the
        // host worker's position is microseconds away, fault-released
        // both ways (the bridge's comment).
        EventHandle const handle = entry.handle;
        EventPool *const pool = entry.pool;
        auto enqueue_wait = [this, consumer, pool, handle] {
            lanes_[consumer].lane->wait_from(pool, handle);
        };
        if (gate_trace_)
        {
            // The gate diagnosis: which host mint family rides this path,
            // and who waits it (kind + plan relation).
            ++gates_total_;
            auto it = host_op_task_.find(ref);
            gate_waited_tasks_[it != host_op_task_.end()
                    ? static_cast<std::uint64_t>(it->second)
                    : 0u] += 1;
            std::size_t const cpos = position;
            int const ckind = static_cast<int>(plan.queue[cpos].kind);
            bool const same_plan = (ref >> 32) == plan.plan_seq;
            gate_consumers_[(same_plan ? 1u : 0u) * 100u +
                static_cast<unsigned>(ckind)] += 1;
        }
        if (bridge_->mint_recorded(ref))
        {
            // the consumer's own stream wait is run-absorbable:
            // the merged bytes of any open COPY run on THIS lane ride
            // behind it (a superset of the unmerged gating — a wait
            // names a strictly earlier op, so the added edge only
            // delays, never cycles).
            std::uint64_t const copy_diag_tsub = copy_diag_copy ? now_ns_local() : 0;
            ++r2_stream_waits_; // census: a stream-wait enqueue
            submit_(consumer, tag, std::move(enqueue_wait), false);
            if (copy_diag_copy)
            {
                copy_diag_.wsub_ns += now_ns_local() - copy_diag_tsub;
            }
        }
        else
        {
            if (consumer < lane_trailing_cache_.size())
            {
                lane_trailing_cache_[consumer] = true;
            }
            std::uint64_t const copy_diag_tsub = copy_diag_copy ? now_ns_local() : 0;
            ++r2_stream_waits_; // census: the parked wait enqueues
                                // the same stream wait at its mark
            bridge_->recorded_wait(
                consumer, tag, ref, std::move(enqueue_wait));
            if (copy_diag_copy)
            {
                copy_diag_.wsub_ns += now_ns_local() - copy_diag_tsub;
            }
        }
    }
    else
    {
        EventHandle const handle = entry.handle;
        EventPool *const pool = entry.pool;
        auto enqueue_wait = [this, consumer, pool, handle] {
            lanes_[consumer].lane->wait_from(pool, handle);
        };
        if (bridge_->mint_recorded(ref))
        {
            // the consumer's own stream wait is run-absorbable:
            // the merged bytes of any open COPY run on THIS lane ride
            // behind it (a superset of the unmerged gating — a wait
            // names a strictly earlier op, so the added edge only
            // delays, never cycles).
            std::uint64_t const copy_diag_tsub = copy_diag_copy ? now_ns_local() : 0;
            ++r2_stream_waits_; // census: a stream-wait enqueue
            submit_(consumer, tag, std::move(enqueue_wait), false);
            if (copy_diag_copy)
            {
                copy_diag_.wsub_ns += now_ns_local() - copy_diag_tsub;
            }
        }
        else
        {
            if (consumer < lane_trailing_cache_.size())
            {
                lane_trailing_cache_[consumer] = true;
            }
            std::uint64_t const copy_diag_tsub = copy_diag_copy ? now_ns_local() : 0;
            ++r2_stream_waits_; // census: the parked wait enqueues
                                // the same stream wait at its mark
            bridge_->recorded_wait(
                consumer, tag, ref, std::move(enqueue_wait));
            if (copy_diag_copy)
            {
                copy_diag_.wsub_ns += now_ns_local() - copy_diag_tsub;
            }
        }
    }
    copy_diag_all_done();
}

// ---- the dispatch  ----

void PlanExecutor::issue_work(
    Plan const &plan, std::size_t position, PlanOp const &op)
{
    switch (op.kind)
    {
    case PlanOpKind::Copy:
    {
        // one occupied-span copy on chan(src.node, dst.node) — the
        // addresses resolved through the nodes' bound addr() surfaces
        // (valid, stable, DMA-stable pool bytes). The byte count
        // is derived HERE from the linked logical
        // chunk's registered tile size — the span the source physical
        // chunk OCCUPIES: min(chunk_class_bytes(), the tile's payload
        // bytes - the chunk's byte offset). A full chunk moves the
        // whole class; a tail chunk moves exactly its payload
        // remainder and nothing more, never the wasted bytes of the
        // block behind it. The tile table aborts loudly on a tile it
        // does not carry (the belt behind the checker's unknown-tile
        // row) and the channel's own guard refuses a degenerate span
        // at submission — zero and over-class are construction
        // faults, never silently clipped.
        LaneId const lane =
            inv_->lane_of_chan(op.u.copy.src.node, op.u.copy.dst.node);
        void const *src = addr_[op.u.copy.src.node](op.u.copy.src.chunk);
        void *dst = addr_[op.u.copy.dst.node](op.u.copy.dst.chunk);
        std::size_t const nbytes =
            tile_chunk_span(tiles_->nbytes(op.u.copy.tile), op.u.copy.index);
        // The COPY-run realization: this op's
        // span JOINS the lane's open run when BOTH endpoints extend it
        // address-exactly (src == run.src + run.nbytes AND dst ==
        // run.dst + run.nbytes — a tail span is never followed by a
        // contiguous address, so interior members are whole-class by
        // arithmetic, never assumption); otherwise the open run
        // flushes and this op opens the next. NOTHING submits here:
        // the run flushes at the lane's next stream-ordering step, at
        // a wait resolving one of its deferred mints, or at the
        // plan's issue end — the merged bytes then ride behind every
        // member's own waits (absorbed inline above) in the same
        // stream order the per-op submits would have produced.
        // On a failed extension of an OPEN run the break census
        // records which endpoint gapped and by how many class blocks
        // (diagnosis only, NNHAUL_COPY_RUNS) — the placement lever's
        // measurement, never control flow.
        CopyRun &run = copy_runs_[lane];
        unsigned const diag_dir =
            inv_->is_host(op.u.copy.dst.node)
                ? (inv_->is_host(op.u.copy.src.node) ? 2u : 0u)
                : (inv_->is_host(op.u.copy.src.node) ? 1u : 2u);
        if (copy_run_diag_on_)
        {
            ++copy_run_diag_.ops[diag_dir];
        }
        // The extension must ALSO stay inside BOTH
        // endpoints' allocations, judged from the RUN'S HEAD: a
        // driver copy may not span two allocations even when they are
        // address-adjacent (invalid argument). contig_
        // reports the one allocation's remaining span from a chunk's
        // base — only the head's span covers the whole extended run
        // (the current member's span would both miss a head in the
        // previous segment and under-count a mid-segment run).
        bool const in_alloc =
            !run.open ||
            (run.nbytes + nbytes <=
                 contig_[op.u.copy.src.node](run.src_chunk) &&
                run.nbytes + nbytes <=
                    contig_[op.u.copy.dst.node](run.dst_chunk));
        if (run.open && in_alloc
            && static_cast<char const *>(run.src) + run.nbytes
                == static_cast<char const *>(src)
            && static_cast<char *>(run.dst) + run.nbytes
                == static_cast<char *>(dst))
        {
            run.nbytes += nbytes;
            run.ops += 1;
            if (copy_run_diag_on_)
            {
                ++copy_run_diag_.merged[diag_dir];
            }
        }
        else
        {
            if (copy_run_diag_on_ && run.open)
            {
                char const *const src_end =
                    static_cast<char const *>(run.src) + run.nbytes;
                char *const dst_end =
                    static_cast<char *>(run.dst) + run.nbytes;
                bool const sx = src_end == src;
                bool const dx = dst_end == dst;
                if (sx && !dx)
                {
                    ++copy_run_diag_.broke_dst[diag_dir];
                }
                else if (dx && !sx)
                {
                    ++copy_run_diag_.broke_src[diag_dir];
                }
                else
                {
                    ++copy_run_diag_.broke_both[diag_dir];
                }
                auto const bump = [](char const *want, char const *end,
                                     std::map<long long, std::uint64_t> &hist,
                                     std::uint64_t &unaligned) {
                    std::ptrdiff_t const delta = want - end;
                    if (delta % static_cast<std::ptrdiff_t>(
                                   chunk_class_bytes())
                        == 0)
                    {
                        ++hist[delta / static_cast<std::ptrdiff_t>(
                                          chunk_class_bytes())];
                        return true; // a class-aligned placement gap
                    }
                    ++unaligned; // a tail span's end: expected,
                                 // never a placement gap
                    return false;
                };
                if (!sx)
                {
                    bool const al = bump(
                        static_cast<char const *>(src), src_end,
                        copy_run_diag_.src_gaps,
                        copy_run_diag_.src_unaligned);
                    if (al)
                    {
                        ++copy_run_diag_.src_al[diag_dir];
                    }
                }
                if (!dx)
                {
                    bool const al = bump(
                        reinterpret_cast<char const *>(dst),
                        reinterpret_cast<char const *>(dst_end),
                        copy_run_diag_.dst_gaps,
                        copy_run_diag_.dst_unaligned);
                    if (al)
                    {
                        ++copy_run_diag_.dst_al[diag_dir];
                    }
                }
            }
            else if (copy_run_diag_on_)
            {
                ++copy_run_diag_.reopened[diag_dir];
            }
            flush_copy_run_(lane);
            run.open = true;
            run.tag = make_op_ref(plan.plan_seq, position);
            run.src = src;
            run.dst = dst;
            run.src_chunk = op.u.copy.src.chunk;
            run.dst_chunk = op.u.copy.dst.chunk;
            run.nbytes = nbytes;
            run.ops = 1;
            run.mints.clear();
        }
        break;
    }
    case PlanOpKind::TileOp:
    {
        // one launch on exec(node) with the operand-address array
        // — engine-owned, kept alive until the queue-close barrier
        // retires it (the host lane's capture contract:
        // launch() copies the POINTER, not the array). The addresses
        // resolve on the engine thread here; the launch call itself
        // rides submit_ so a trailing lane keeps its order. The
        // operand pool: one bump over the slot's flat arena — no
        // per-op allocation, and the exact-capacity reserve keeps
        // every taken array's address stable for the plan's life
        // (the slot retires at the close, as the contract demands).
        LaneId const lane = inv_->lane_of_exec(op.u.tile_op.node);
        std::size_t const n = op.operands.size();
        if (cur_->operand_pool_used + n > cur_->operand_pool.size())
        {
            cur_->operand_pool.resize(cur_->operand_pool_used + n);
        }
        void **ptr = cur_->operand_pool.data() + cur_->operand_pool_used;
        cur_->operand_pool_used += n;
        for (std::size_t i = 0; i < n; ++i)
        {
            Operand const &o = op.operands[i];
            ptr[i] = addr_[o.node](o.chunk);
        }
        TaskId const task = op.u.tile_op.task;
        NodeId const node = op.u.tile_op.node;
        submit_(lane, make_op_ref(plan.plan_seq, position),
            [this, lane, task, node, ptr] {
                lanes_[lane].lane->launch(task, node, ptr);
            });
        break;
    }
    case PlanOpKind::Link:
    case PlanOpKind::Unlink:
        // Unreachable — the loop routes laneless ops before the
        // dispatch; a LINK/UNLINK reaching here is an engine defect.
        throw std::runtime_error(
            "plan executor: a laneless op reached the dispatch — LINK "
            "and UNLINK are binding-table transitions only");
    }
}

// ---- COPY fusion: why the memcpy batch exists and the kernel arm does not
//
// Same-lane COPY runs may realize as one descriptor launch. Three arms
// were evaluated: a one-launch chunked copy kernel over a per-channel
// device pointer ring, an N-call memcpy
// batch, and the unfused path. On a saturated card the device copy kernel
// steals compute SMs and regresses visibly on a saturated card; the
// memcpy batch stays within noise of unfused and cuts driver-lock
// contention. Consecutive same-lane COPY ops whose spans are
// address-contiguous at BOTH endpoints merge into ONE cudaMemcpyAsync:
// plan content, op order, the issue log, per-op stamps, trace events and
// every byte accounting are untouched; waits absorb, mints defer, and the
// run flushes at the lane's next ordering step (plan_executor.hh carries
// the soundness argument). No device copy kernel exists in this tree.

// ---- the pass, act one: ISSUE (execute(plan P) through the
//      issue loop — everything but the queue close) ----

std::size_t PlanExecutor::issue(Plan const &plan)
{
    // The enqueue-before-derive calling discipline, demanded loudly:
    // plans issue in run-queue order, consecutive from cold start —
    // plan k+1's derivation reads facts describing a fully enqueued
    // prefix only if every earlier plan of the run went through THIS
    // executor, in order. The barriers are NOT part of
    // the discipline: a later plan issues while an earlier one still
    // runs; the discipline is the ENQUEUE
    // order, and the single submission path keeps it.
    if (!have_seq_ && plan.plan_seq != 0u)
    {
        abort_census(plan, kNoPosition,
            "plan-seq-discontinuity: the run's first plan carries "
            "plan_seq " +
                std::to_string(plan.plan_seq) +
                ", not 0 — the run queue starts at cold start (the "
                "engine's enqueue-before-derive discipline)");
    }
    if (have_seq_ && plan.plan_seq != last_seq_ + 1u)
    {
        abort_census(plan, kNoPosition,
            "plan-seq-discontinuity: plan_seq " +
                std::to_string(plan.plan_seq) + " follows plan_seq " +
                std::to_string(last_seq_) +
                " — plans issue in run-queue order, consecutive from "
                "cold start (the engine's enqueue-before-derive "
                "discipline)");
    }

    // (0) The boundary check — the entry-state half of planned ==
    // issued, over ALL nodes' segments, binding for binding, flag for
    // flag, BEFORE any op issues. The
    // applied table advanced through every earlier plan's TRANSITIONS
    // at their issue instants, so it equals the earlier plan's
    // predicted exit — this plan's entry — whatever barriers stand.
    if (!applied_.equals(plan.entry))
    {
        abort_census(plan, kNoPosition,
            "boundary-mismatch: the applied binding table differs from "
            "the plan's entry state over the nodes' segments");
    }
    // Growth is a COMPILATION act:
    // the plan's growth (its final cardinalities) must already be
    // committed — pools grown, inventory/applied/dependency extended —
    // BEFORE this executor issues anything. The run phase executes
    // exactly what is ordered; it never allocates. A plan naming a
    // cardinality the inventory does not hold is a caller defect
    // (apply_plan_growth was skipped), loud here.
    for (std::size_t n = 0; n < plan.growth.size(); ++n)
    {
        if (plan.growth[n] != 0
            && inv_->pool_cardinality(static_cast<NodeId>(n))
                < plan.growth[n])
        {
            abort_census(plan, kNoPosition,
                "growth-not-committed: the plan grows node " +
                std::to_string(n) + " to " +
                std::to_string(plan.growth[n]) +
                " chunks but the inventory holds " +
                std::to_string(inv_->pool_cardinality(
                    static_cast<NodeId>(n))) +
                " — growth belongs to the compilation phase "
                "(apply_plan_growth), never the run phase");
        }
    }

    // The retained slot for this plan's issue state (the depth-2
    // shape: its buffers stand until THIS plan's close).
    std::size_t slot_index;
    if (!free_slots_.empty())
    {
        slot_index = free_slots_.back();
        free_slots_.pop_back();
        slots_[slot_index] = std::make_unique<Slot>();
    }
    else
    {
        slot_index = slots_.size();
        slots_.push_back(std::make_unique<Slot>());
    }
    Slot &slot = *slots_[slot_index];
    cur_ = &slot;
    struct CurGuard
    {
        PlanExecutor *ex;
        ~CurGuard()
        {
            ex->cur_ = nullptr;
        }
    } cur_guard{this};

    // The tracer's registration (engine thread, before issue); a
    // disarmed tracer places nothing for it.
    if (tracer_ != nullptr)
    {
        tracer_->begin_plan(plan);
    }

    // The per-plan issue state.
    slot.plan_seq = plan.plan_seq;
    slot.issue_log.clear();
    slot.issued_waits.assign(plan.queue.size(), {});
    slot.mint_handles.assign(plan.queue.size(), 0);
    slot.stamp_refs.assign(
        plan.queue.size(), StampRef{}); // placed == false: nothing placed
    slot.tail_stamps.assign(inv_->lane_count(), 0);
    slot.tail_placed.assign(inv_->lane_count(), 0);
    slot.terminal_handles.assign(inv_->lane_count(), 0);
    slot.host_flags.clear();
    slot.host_flag_args.clear();
    slot.last_op_on_lane.assign(inv_->lane_count(), kNoPosition);

    // The elision's per-plan state: the per-(consumer,owner)
    // epoch high-water empties (a wait of THIS plan can never be
    // covered by a previous plan's waits — the first wait per pair
    // always enqueues), and the per-plan census zeroes.
    std::fill(wait_hw_.begin(), wait_hw_.end(), 0);
    k9_elided_ = 0;
    k9_waited_ = 0;
    k9_fast_elided_ = 0;

    // The operand pool's per-plan reset: the operand pool bumps
    // from zero (one exact-capacity reserve keeps every taken array's
    // address stable for the plan's life — the slot retires at close).
    slot.operand_pool_used = 0;
    std::size_t operand_total = 0;
    for (PlanOp const &op : plan.queue)
    {
        if (op.kind == PlanOpKind::TileOp)
        {
            operand_total += op.operands.size();
        }
    }
    if (slot.operand_pool.capacity() < operand_total)
    {
        slot.operand_pool.reserve(operand_total);
    }

    // (1) The derivation pass — the one dependency pass,
    // its state already carrying every earlier plan's facts — also
    // the ones still RUNNING: their enqueues are complete (the
    // discipline above), so every fact this pass reads describes a
    // fully enqueued prefix. The derivation-time enqueues (late
    // records, host-flag notifies) ride submit_ — the single issue
    // order — so a bridge still trailing an earlier plan's steps
    // keeps that order; this is exactly the case the machinery
    // exists for (a successor deriving while the predecessor's lanes
    // still run).
    bool const dbg = std::getenv("NNHAUL_ISSUE_BREAKDOWN") != nullptr;
    std::uint64_t const t_derive0 = dbg ? now_ns_local() : 0;
    // The segment executor's derive: consecutive same-lane ops
    // group into one segment and the intra-segment domination drop
    // removes owed refs provably covered by an earlier member's kept
    // wait (dependency.hh's note) — the runtime wait elision made
    // structural, the enqueued wait set bit-identical.
    slot.deriv = derivation_.run(plan, true);
    last_derivation_ = slot.deriv;
    PlanDerivation &deriv = slot.deriv;
    std::uint64_t const ns_derive =
        dbg ? now_ns_local() - t_derive0 : 0;

    // The host-flag needs first: every ref a HOST-lane op owes gets a
    // host flag this plan gates on, acquired once per
    // waited mint whatever the mint's plan of origin.
    for (std::size_t p = 0; p < plan.queue.size(); ++p)
    {
        if (lane_of_op(plan.queue[p], *inv_) != host_exec_lane_)
        {
            continue;
        }
        for (OpRef a : deriv.owed_set[p])
        {
            if (slot.host_flags.find(a) == slot.host_flags.end())
            {
                slot.host_flags.emplace(a, acquire_host_flag());
            }
        }
    }

    // The mint acquisitions, late records first: a late record's event
    // mints on the OWED op's lane, and the record is ENQUEUED into
    // that lane at THIS point — strictly after the owed op's device
    // work by enqueue-before-derive: the late-record placement. The
    // record may land in a LIVE
    // lane — an earlier, un-joined plan's lane mid-execution — sound
    // by the in-order lane (the record trails its tail). (The
    // discovery-order submission stayed: a ref-ascending sort built
    // for the rejected wait elision bought nothing and was reverted.)
    for (LateRecord const &lr : deriv.late_records)
    {
        EventPool *const home = lr.owner == host_exec_lane_
            ? host_mint_pool_
            : lanes_[lr.owner].pool;
        EventHandle const handle = home->acquire();
        ++mints_acquired_;
        registry_.insert(
            lr.op, EventRegistry::Entry{handle, lr.owner, home});
        try
        {
            // the record-submission epoch stamp, at the submit_
            // call (engine-thread order = the owner lane's stream
            // order; see plan_executor.hh's elision note).
            stamp_rec_(lr.op, lr.owner);
            submit_(lr.owner, lr.op, [this, owner = lr.owner, handle,
                                         op = lr.op] {
                mint_with_twin_(owner, handle, op);
            });
            if (slot.host_flags.find(lr.op) != slot.host_flags.end())
            {
                enqueue_host_flag_notify(lr.owner, lr.op);
            }
        }
        catch (std::exception const &ex)
        {
            abort_census(plan, kNoPosition,
                "issue-abort: the late record of " + render_ref(lr.op) +
                    " into lane " + inv_->lane_name(lr.owner) +
                    " was refused: " + ex.what());
        }
    }

    // Then this plan's own mints: acquired and registered now, the
    // record enqueued at the op's issue position immediately after its
    // device work.
    std::uint64_t const t_mint0 = dbg ? now_ns_local() : 0;
    for (std::size_t p = 0; p < plan.queue.size(); ++p)
    {
        if (!deriv.mints[p])
        {
            continue;
        }
        LaneId const lane = lane_of_op(plan.queue[p], *inv_);
        // A HOST-lane op's mint is a TWIN (a device-pool event the
        // host worker records at the op's FIFO position); every other
        // lane mints from its own family pool as always.
        EventPool *const home =
            lane == host_exec_lane_ ? host_mint_pool_ : lanes_[lane].pool;
        EventHandle const handle = home->acquire();
        ++mints_acquired_;
        slot.mint_handles[p] = handle;
        registry_.insert(make_op_ref(plan.plan_seq, p),
            EventRegistry::Entry{handle, lane, home});
    }
    std::uint64_t const ns_mintacq =
        dbg ? now_ns_local() - t_mint0 : 0;

    // Pre-existing mints first waited by a host consumer of THIS plan:
    // the flag-setting notify goes into the producer lane now — after
    // its whole enqueued tail, which includes the mint (conservative,
    // sound by the in-order lane, with the same pricing the late record
    // carries).
    for (auto const &need : slot.host_flags)
    {
        OpRef const producer = need.first;
        bool const ours = (producer >> 32) == plan.plan_seq &&
            deriv.mints[static_cast<std::size_t>(producer & 0xffffffffu)];
        bool const late = std::any_of(deriv.late_records.begin(),
            deriv.late_records.end(),
            [&producer](LateRecord const &lr) { return lr.op == producer; });
        if (ours || late)
        {
            continue; // the notify rides the mint's own enqueue
        }
        EventRegistry::Entry entry = registry_.resolve(producer);
        try
        {
            enqueue_host_flag_notify(entry.owner, producer);
        }
        catch (std::exception const &ex)
        {
            abort_census(plan, kNoPosition,
                "issue-abort: the host-flag notify of " +
                    render_ref(producer) + " into lane " +
                    inv_->lane_name(entry.owner) +
                    " was refused: " + ex.what());
        }
    }

    // The documented test seam: between derive and issue.
    if (post_derivation_hook_)
    {
        post_derivation_hook_(plan, deriv);
    }

    // (2) The loop over P.queue in order — the straight line whose
    // only loops are the queue and the lane set.
    // (The breakdown knob's brackets: diagnosis-only steady-clock
    // reads around the loop's four cost centers — the laneless
    // transitions, the derived waits, the dispatched work, and the
    // rest of the per-op body — env-gated like the old engine's
    // NNHAUL_ISSUE_BREAKDOWN, off by default, never control flow.
    // The fine split adds per-KIND centers on a second line:
    // resolve+rows (lane_of_op, light_rows), waits (issue_wait),
    // submit (the dispatch/mint/notify submit_ calls), transition
    // (apply_transition), and log (the issue-log and wait-log
    // appends) — the same diagnosis-only discipline.)
    bool const breakdown =
        std::getenv("NNHAUL_ISSUE_BREAKDOWN") != nullptr;
    // The boundary brackets (diagnosis only, never control flow):
    // absolute steady-clock ns at the issue pass's own walls — entry,
    // the issue loop's first dispatch, and the loop end after the
    // terminal mints — read by the boundary drain-tail decode.
    bool const boundary = boundary_breakdown_on();
    std::uint64_t const tb_issue_entry = boundary ? now_ns_local() : 0;
    std::uint64_t tb_loop0 = 0, tb_loop_end = 0;
    std::uint64_t ns_laneless = 0, ns_waits = 0, ns_work = 0,
                   ns_rest = 0;
    // The fine split: [kind][center], centers are resolve+rows,
    // waits, submit, transition, log; plus per-kind op counts.
    enum FineCenter
    {
        kFineResolve = 0,
        kFineWait,
        kFineSubmit,
        kFineTrans,
        kFineLog,
        kFineCenters
    };
    std::uint64_t fine_ns[4][kFineCenters] = {};
    std::uint64_t fine_ops[4] = {0, 0, 0, 0};
    tb_loop0 = boundary ? now_ns_local() : 0;
    for (std::size_t p = 0; p < plan.queue.size(); ++p)
    {
        PlanOp const &op = plan.queue[p];
        OpRef const ref = make_op_ref(plan.plan_seq, p);
        std::uint64_t const tb = breakdown ? now_ns_local() : 0;
        std::uint64_t op_wait_ns = 0;

        // The lane: a pure function of the op's own fields.
        // A TILE_OP naming an unknown node throws here (the
        // unknown-id row's belt); a COPY naming a channel the
        // inventory lacks carries no lane and dies at the rows below
        // (copy-no-channel).
        LaneId lane = kNoLane;
        try
        {
            lane = lane_of_op(op, *inv_);
        }
        catch (std::exception const &ex)
        {
            abort_census(plan, p, std::string("issue-abort: ") + ex.what());
        }

        // The rows re-evaluated abort-only (the placement note of
        // light_rows: pre-op state, rows before effects).
        std::uint64_t const copy_diag_rows0 =
            copy_diag_on_ && op.kind == PlanOpKind::Copy ? now_ns_local() : 0;
        std::string const row = light_rows(op);
        if (copy_diag_on_ && op.kind == PlanOpKind::Copy)
        {
            copy_diag_.rows_ns += now_ns_local() - copy_diag_rows0;
            ++copy_diag_.copy_ops;
        }
        std::uint64_t const t_resolve =
            breakdown ? now_ns_local() : 0;
        unsigned const kind_idx = static_cast<unsigned>(op.kind);
        if (breakdown)
        {
            fine_ns[kind_idx][kFineResolve] += t_resolve - tb;
        }
        if (!row.empty())
        {
            abort_census(plan, p,
                "issue-abort: the re-evaluated row fired — " + row);
        }

        // LINK and UNLINK: the binding-table transition IS the whole
        // op (no backend call, no lane, no event, no stamps).
        if (lane == kNoLane)
        {
            try
            {
                apply_transition(op);
            }
            catch (std::exception const &ex)
            {
                abort_census(plan, p,
                    std::string("issue-abort: the applied table refused "
                                "the transition: ") +
                        ex.what());
            }
            slot.issue_log.push_back(op);
            ++ops_issued_;
            if (breakdown)
            {
                std::uint64_t const tl = now_ns_local();
                ns_laneless += tl - tb;
                fine_ops[kind_idx] += 1;
                fine_ns[kind_idx][kFineTrans] += tl - t_resolve;
                fine_ns[kind_idx][kFineLog] += now_ns_local() - tl;
            }
            continue;
        }

        // Derived waits first — before the op's own device work.
        // A refusal aborts immediately: dependents never
        // run, nothing further is enqueued on any lane.
        for (OpRef a : deriv.owed_set[p])
        {
            try
            {
                issue_wait(plan, p, lane, a);
            }
            catch (ExecutorCensusError const &)
            {
                throw;
            }
            catch (std::exception const &ex)
            {
                std::string const detail = ex.what();
                if (detail.rfind("registry-miss", 0) == 0)
                {
                    abort_census(plan, p,
                        detail + ", waited by op " + op_line(plan, p));
                }
                abort_census(plan, p,
                    "issue-abort: the wait of " + render_ref(a) +
                        " was refused: " + detail);
            }
            slot.issued_waits[p].push_back(a);
        }
        if (breakdown)
        {
            op_wait_ns = now_ns_local() - tb;
            ns_waits += op_wait_ns;
            fine_ns[kind_idx][kFineWait] += op_wait_ns -
                (t_resolve - tb);
        }
        std::uint64_t const t_wait_end =
            breakdown ? tb + op_wait_ns : 0;

        // The tracer's stamps — armed with the tracer only (an
        // untraced run places none of them). ONE stamp
        // per instrumented op: the clock read at the op's ISSUE
        // position on its own lane (the submission into the lane is
        // in the lane's order by construction — the read is AT the
        // lane position whatever thread submits, the engine thread
        // direct or the bridge's waiter; the host lane's worker reads
        // its own clock at the FIFO position). The
        // op's measured interval is the TILE it owns of the lane's
        // issue order — [its stamp, the next stamp on that lane] —
        // composed at THIS plan's close, after the barriers — facts,
        // never inputs: stamps gate nothing.
        bool const traced = tracer_ != nullptr && tracer_->armed();
        EventHandle stamp_read_handle = 0;
        std::uint64_t const copy_diag_stamp0 =
            copy_diag_on_ && traced && op.kind == PlanOpKind::Copy
            ? now_ns_local()
            : 0;
        try
        {
            if (traced)
            {
                stamp_read_handle = lanes_[lane].lane->stamp_acquire();
                EventHandle const sb = stamp_read_handle;
                // the stamp is a host clock read — zero CUDA API,
                // no stream step — so it passes an open COPY run of
                // the lane without closing it (the measured tile
                // boundaries are facts; the merged bytes' enqueue
                // instant is a fact that may differ from the unmerged
                // run's, exactly the stamp rule's allowance).
                submit_(lane, ref, [this, lane, sb] {
                    lanes_[lane].lane->stamp(sb);
                }, false);
            }
            if (copy_diag_on_ && traced && op.kind == PlanOpKind::Copy)
            {
                copy_diag_.stamp_ns += now_ns_local() - copy_diag_stamp0;
                ++copy_diag_.stamp_n;
            }

            // The dispatch.
            issue_work(plan, p, op);

            if (traced)
            {
                slot.stamp_refs[p] =
                    StampRef{lane, stamp_read_handle, true, p};
            }

            // The mint immediately after the work,
            // then — when a host consumer gates on this mint — the
            // flag-setting notify in this same lane right after the
            // record.
            if (deriv.mints[p])
            {
                EventHandle const handle = slot.mint_handles[p];
                // A COPY-run member's mint DEFERS into its run:
                // the record rides the flush — after the merged bytes,
                // in issue order — the same stream position relative
                // to the member's own work the unmerged record held
                // (consumers resolve it through issue_wait's flush
                // hook, the capture rule intact). A mint carrying a
                // host-flag notify never defers (the notify's enqueue
                // position is the flag's contract), and neither does
                // any non-COPY lane step: the submit below closes the
                // run first, bytes before record before notify.
                if (op.kind == PlanOpKind::Copy
                    && copy_runs_[lane].open
                    && slot.host_flags.find(ref)
                        == slot.host_flags.end())
                {
                    copy_runs_[lane].mints.emplace_back(ref, handle);
                }
                else
                {
                    // the epoch stamp at the submit_ call (the
                    // deferred arm above stamps at the run's flush,
                    // in member order).
                    stamp_rec_(ref, lane);
                    submit_(lane, ref, [this, lane, handle, ref] {
                        mint_with_twin_(lane, handle, ref);
                    });
                    if (slot.host_flags.find(ref)
                        != slot.host_flags.end())
                    {
                        enqueue_host_flag_notify(lane, ref);
                    }
                }
            }
        }
        catch (std::exception const &ex)
        {
            // The census wrap: the lane's own fault text carries the device;
            // the census naming the op is the executor's artifact (the
            // lane cannot know the OpRef).
            abort_census(plan, p,
                std::string("issue-abort: the op's enqueue was refused: ") +
                    ex.what());
        }
        if (breakdown)
        {
            std::uint64_t const tw = now_ns_local();
            ns_work += tw - tb - op_wait_ns;
            fine_ns[kind_idx][kFineSubmit] += tw - t_wait_end;
            fine_ops[kind_idx] += 1;
            op_wait_ns = tw - tb; // this op's total so far
        }

        slot.last_op_on_lane[lane] = p;
        if (gate_trace_ && lane == host_exec_lane_ &&
            op.kind == PlanOpKind::TileOp)
        {
            host_op_task_[ref] = op.u.tile_op.task;
        }

        // The applied table advances by the same fold transition.
        try
        {
            apply_transition(op);
        }
        catch (std::exception const &ex)
        {
            abort_census(plan, p,
                std::string("issue-abort: the applied table refused the "
                            "transition: ") +
                    ex.what());
        }
        std::uint64_t const t_trans_end =
            breakdown ? now_ns_local() : 0;
        if (breakdown)
        {
            fine_ns[kind_idx][kFineTrans] += t_trans_end - (tb +
                op_wait_ns);
        }

        slot.issue_log.push_back(op);
        ++ops_issued_;
        if (breakdown)
        {
            ns_rest += t_trans_end - tb - op_wait_ns;
            fine_ns[kind_idx][kFineLog] += now_ns_local() -
                t_trans_end;
        }
    }

    // Every open COPY run flushes at the issue's end — plan-local
    // by construction, so no run survives into a successor's issue
    // (the depth-2 discipline: the successor's late records and
    // notifies step these lanes at their own issue positions, behind
    // this plan's bytes). The tail stamps and terminal mints below
    // submit after the flush, closing it anyway; this is the belt.
    flush_all_copy_runs_();
    if (copy_diag_on_)
    {
        // The per-COPY executor-path split, one line per plan
        // (diagnosis only): the four cost centers the cell delta must
        // live in — (i) the wait walk (resolution apart from the
        // enqueue that carries the inline cudaStreamWaitEvent), (ii)
        // the flush's merged memcpy submit, (iii) the deferred member
        // mint records, (iv) the stamps — plus the census counts (the
        // same-lane leak and the flush-hook hits).
        std::fprintf(stderr,
            "[nnhaul-k5] seq=%llu copies=%llu wait_n=%llu wait_us=%.2f "
            "wres_us=%.2f wsub_us=%.2f samelane=%llu hook=%llu "
            "flush_n=%llu flprep_us=%.2f flcopy_us=%.2f mrec_n=%llu "
            "mrec_us=%.2f stamp_n=%llu stamp_us=%.2f rows_us=%.2f "
            "twait_n=%llu twait_us=%.2f twres_us=%.2f\n",
            static_cast<unsigned long long>(plan.plan_seq),
            static_cast<unsigned long long>(copy_diag_.copy_ops),
            static_cast<unsigned long long>(copy_diag_.wait_n),
            static_cast<double>(copy_diag_.wait_ns) / 1.0e3,
            static_cast<double>(copy_diag_.wres_ns) / 1.0e3,
            static_cast<double>(copy_diag_.wsub_ns) / 1.0e3,
            static_cast<unsigned long long>(copy_diag_.samelane),
            static_cast<unsigned long long>(copy_diag_.flushhook),
            static_cast<unsigned long long>(copy_diag_.flush_n),
            static_cast<double>(copy_diag_.flprep_ns) / 1.0e3,
            static_cast<double>(copy_diag_.flcopy_ns) / 1.0e3,
            static_cast<unsigned long long>(copy_diag_.mrec_n),
            static_cast<double>(copy_diag_.mrec_ns) / 1.0e3,
            static_cast<unsigned long long>(copy_diag_.stamp_n),
            static_cast<double>(copy_diag_.stamp_ns) / 1.0e3,
            static_cast<double>(copy_diag_.rows_ns) / 1.0e3,
            static_cast<unsigned long long>(copy_diag_.tilo_wait_n),
            static_cast<double>(copy_diag_.tilo_wait_ns) / 1.0e3,
            static_cast<double>(copy_diag_.tilo_wres_ns) / 1.0e3);
        // The elision census, one line per plan (diagnosis only,
        // same knob as the k5 split): waits elided behind the epoch
        // high-water (fast = resolved through the dense rows without
        // the registry hash) apart from waits enqueued.
        std::fprintf(stderr,
            "[nnhaul-k9] seq=%llu elided=%llu fast=%llu waited=%llu "
            "next_epoch=%llu\n",
            static_cast<unsigned long long>(plan.plan_seq),
            static_cast<unsigned long long>(k9_elided_),
            static_cast<unsigned long long>(k9_fast_elided_),
            static_cast<unsigned long long>(k9_waited_),
            static_cast<unsigned long long>(next_rec_epoch_));
        // The segment census, one line per plan (diagnosis only,
        // same knob): the segments the derivation grouped, the ops
        // they cover, the raw per-op folded refs vs the refs the
        // intra-segment domination drop kept (the difference = the
        // dominated edges that never materialize as wait walks), the
        // mints + late records, and the driver-class call counts
        // (mint-record submits, stream-wait enqueues).
        std::fprintf(stderr,
            "[nnhaul-r2] seq=%llu segments=%zu members=%zu raw=%zu "
            "head=%zu folded=%zu mints=%zu late=%zu rec_submits=%llu "
            "stream_waits=%llu\n",
            static_cast<unsigned long long>(plan.plan_seq),
            deriv.segments,
            deriv.segment_ops,
            deriv.raw_refs,
            deriv.head_refs,
            deriv.raw_refs - deriv.head_refs,
            static_cast<std::size_t>(std::count(deriv.mints.begin(),
                deriv.mints.end(), true)),
            deriv.late_records.size(),
            static_cast<unsigned long long>(r2_rec_submits_),
            static_cast<unsigned long long>(r2_stream_waits_));
        r2_rec_submits_ = 0;
        r2_stream_waits_ = 0;
        copy_diag_ = CopyDiag{};
    }
    if (copy_run_diag_on_ && copy_run_ops_ != 0)
    {
        std::fprintf(stderr,
            "[nnhaul-copy-runs] seq=%llu run_ops=%llu calls_saved=%llu\n",
            static_cast<unsigned long long>(plan.plan_seq),
            static_cast<unsigned long long>(copy_run_ops_),
            static_cast<unsigned long long>(copy_run_calls_saved_));
        // The break census line: per direction (d2h/h2d/d2d),
        // ops/merged/src-broke/dst-broke/both-broke/reopened, then the
        // gapped endpoints' signed class-block distances (top four by
        // count; "u" = unaligned tail ends, expected). Names the side
        // a placement fix must make contiguous.
        static char const *kDirName[3] = {"d2h", "h2d", "d2d"};
        std::fprintf(stderr, "[nnhaul-copy-brk] seq=%llu",
            static_cast<unsigned long long>(plan.plan_seq));
        for (unsigned d = 0; d < 3; ++d)
        {
            if (copy_run_diag_.ops[d] == 0)
            {
                continue;
            }
            std::fprintf(stderr,
                " %s ops=%llu mg=%llu s=%llu d=%llu b=%llu sa=%llu "
                "da=%llu re=%llu",
                kDirName[d],
                static_cast<unsigned long long>(copy_run_diag_.ops[d]),
                static_cast<unsigned long long>(
                    copy_run_diag_.merged[d]),
                static_cast<unsigned long long>(
                    copy_run_diag_.broke_src[d]),
                static_cast<unsigned long long>(
                    copy_run_diag_.broke_dst[d]),
                static_cast<unsigned long long>(
                    copy_run_diag_.broke_both[d]),
                static_cast<unsigned long long>(
                    copy_run_diag_.src_al[d]),
                static_cast<unsigned long long>(
                    copy_run_diag_.dst_al[d]),
                static_cast<unsigned long long>(
                    copy_run_diag_.reopened[d]));
        }
        auto const render_gaps =
            [](char const *side,
               std::map<long long, std::uint64_t> const &hist,
               std::uint64_t unaligned) {
                std::fprintf(stderr, " %sgaps", side);
                std::vector<std::pair<std::uint64_t, long long>> top;
                for (std::pair<long long const, std::uint64_t> const &e :
                    hist)
                {
                    top.emplace_back(e.second, e.first);
                }
                std::sort(top.begin(), top.end(),
                    [](std::pair<std::uint64_t, long long> const &a,
                        std::pair<std::uint64_t, long long> const &b) {
                        return a.first > b.first;
                    });
                std::size_t const n =
                    top.size() < 4 ? top.size() : 4;
                for (std::size_t i = 0; i < n; ++i)
                {
                    std::fprintf(stderr, " %lld:%llu", top[i].second,
                        static_cast<unsigned long long>(top[i].first));
                }
                if (unaligned != 0)
                {
                    std::fprintf(stderr, " u:%llu",
                        static_cast<unsigned long long>(unaligned));
                }
            };
        render_gaps("src", copy_run_diag_.src_gaps,
            copy_run_diag_.src_unaligned);
        render_gaps("dst", copy_run_diag_.dst_gaps,
            copy_run_diag_.dst_unaligned);
        std::fprintf(stderr, "\n");
        copy_run_ops_ = 0;
        copy_run_calls_saved_ = 0;
        copy_run_diag_ = CopyRunDiag{};
    }

    // THE ISSUE ORDER'S TAIL STAMPS: one more
    // clock read per touched lane, at the issue's end — the LAST
    // tile's end on that lane (the tile the plan's last op owns of
    // the lane's issue order runs to here). Rides submit_ (the
    // single issue order), like every stamp.
    if (tracer_ != nullptr && tracer_->armed())
    {
        try
        {
            for (LaneId lane = 0; lane < inv_->lane_count(); ++lane)
            {
                if (slot.last_op_on_lane[lane] == kNoPosition)
                {
                    continue; // untouched lane: no tiles, no tail
                }
                EventHandle const tail = lanes_[lane].lane->stamp_acquire();
                submit_(lane, make_op_ref(plan.plan_seq, plan.queue.size()),
                    [this, lane, tail] { lanes_[lane].lane->stamp(tail); });
                slot.tail_stamps[lane] = tail;
                slot.tail_placed[lane] = 1;
            }
        }
        catch (std::exception const &ex)
        {
            abort_census(plan, kNoPosition,
                std::string("issue-abort: the issue-order tail stamp "
                            "was refused: ") +
                    ex.what());
        }
    }

    // THE TERMINAL MINTS (the previous engine's batch-granular join,
    // ported): at the issue's
    // end, every lane this plan touched records one event at its
    // tail — the close joins THAT EVENT, never the stream, so the
    // successor's already-fed prefix never delays the join. The mint
    // rides submit_ (the single issue order): it is this plan's LAST
    // step on the lane whatever the path, direct or deferred.
    for (LaneId lane = 0; lane < inv_->lane_count(); ++lane)
    {
        if (lane == host_exec_lane_ ||
            slot.last_op_on_lane[lane] == kNoPosition)
        {
            continue; // untouched lane: nothing to join
        }
        EventHandle const handle = lanes_[lane].pool->acquire();
        submit_(lane, make_op_ref(plan.plan_seq, plan.queue.size()),
            [this, lane, handle] { lanes_[lane].lane->mint(handle); });
        slot.terminal_handles[lane] = handle;
    }
    tb_loop_end = boundary ? now_ns_local() : 0;

    if (breakdown)
    {
        std::fprintf(stderr,
            "[nnhaul-issue-loop] seq=%llu ops=%zu laneless_ms=%.2f "
            "waits_ms=%.2f work_ms=%.2f rest_ms=%.2f\n",
            static_cast<unsigned long long>(plan.plan_seq),
            plan.queue.size(),
            static_cast<double>(ns_laneless) / 1.0e6,
            static_cast<double>(ns_waits) / 1.0e6,
            static_cast<double>(ns_work) / 1.0e6,
            static_cast<double>(ns_rest) / 1.0e6);
        // The fine split (diagnosis only): per kind, mean
        // MICROSECONDS per op of each center, so a capped-vs-uncapped
        // diff names the center the marginal cost lives in. Centers:
        // res (lane+rows), wait, sub (submit_ calls incl. the driver
        // enqueue), tr (apply_transition), log (issue/wait log
        // appends); n = ops of the kind.
        char const *const names[4] = {"LINK", "UNLK", "COPY", "TILO"};
        for (unsigned k = 0; k < 4; ++k)
        {
            if (fine_ops[k] == 0)
            {
                continue;
            }
            double const inv = 1.0 /
                static_cast<double>(fine_ops[k]); // us per op
            std::fprintf(stderr,
                "[nnhaul-issue-split] seq=%llu kind=%s n=%llu "
                "res_us=%.2f wait_us=%.2f sub_us=%.2f tr_us=%.2f "
                "log_us=%.2f\n",
                static_cast<unsigned long long>(plan.plan_seq),
                names[k],
                static_cast<unsigned long long>(fine_ops[k]),
                static_cast<double>(fine_ns[k][kFineResolve]) / 1.0e3 * inv,
                static_cast<double>(fine_ns[k][kFineWait]) / 1.0e3 * inv,
                static_cast<double>(fine_ns[k][kFineSubmit]) / 1.0e3 * inv,
                static_cast<double>(fine_ns[k][kFineTrans]) / 1.0e3 * inv,
                static_cast<double>(fine_ns[k][kFineLog]) / 1.0e3 * inv);
        }
        std::fprintf(stderr,
            "[nnhaul-issue-pre] seq=%llu derive_ms=%.2f mint_ms=%.2f "
            "mint_n=%zu late_n=%zu\n",
            static_cast<unsigned long long>(plan.plan_seq),
            static_cast<double>(ns_derive) / 1.0e6,
            static_cast<double>(ns_mintacq) / 1.0e6,
            static_cast<std::size_t>(std::count(deriv.mints.begin(),
                deriv.mints.end(), true)),
            deriv.late_records.size());
    }

    have_seq_ = true;
    last_seq_ = plan.plan_seq;
    if (boundary)
    {
        std::fprintf(stderr,
            "[nnhaul-boundary-exe] seq=%llu t_entry=%llu t_loop0=%llu "
            "t_loop_end=%llu\n",
            static_cast<unsigned long long>(plan.plan_seq),
            static_cast<unsigned long long>(tb_issue_entry),
            static_cast<unsigned long long>(tb_loop0),
            static_cast<unsigned long long>(tb_loop_end));
    }
    return slot_index;
}

// ---- the pass, act two: CLOSE  ----

void PlanExecutor::close(Plan const &plan,
    std::size_t slot_index)
{
    if (slot_index >= slots_.size() || slots_[slot_index] == nullptr ||
        slots_[slot_index]->plan_seq != plan.plan_seq)
    {
        throw std::runtime_error(
            "plan executor: close() of a slot that is not the named "
            "plan's retained slot — the engine's slot discipline (the "
            "join act)");
    }
    cur_ = slots_[slot_index].get();
    struct CurGuard
    {
        PlanExecutor *ex;
        ~CurGuard()
        {
            ex->cur_ = nullptr;
        }
    } cur_guard{this};

    // The full PlanIssue witness now lives in last_issue() —
    // close() itself no longer hands a copy to the caller (the deep
    // copy per close was a measurable tail; engine_note_batch reads
    // last_issue() exactly as the fused execute() path always has).
    close_queue(plan, slots_[slot_index]->deriv, *slots_[slot_index]);

    slots_[slot_index].reset(); // the retained state retires
    free_slots_.push_back(slot_index);
}

PlanDerivation const &PlanExecutor::slot_derivation(
    std::size_t slot_index) const
{
    if (slot_index >= slots_.size() || slots_[slot_index] == nullptr)
    {
        throw std::runtime_error(
            "plan executor: slot_derivation() of a retired slot");
    }
    return slots_[slot_index]->deriv;
}

// ---- the queue close (close_queue — the JOIN act) ----

void PlanExecutor::close_queue(Plan const &plan,
    PlanDerivation const &deriv, Slot &slot)
{
    std::uint64_t const released_at_entry = handles_released_;

    // The boundary brackets (diagnosis only, never control flow):
    // absolute walls at the close's OWN sub-phases — entry/reconcile,
    // then the join/host/sweep/harvest walls, with the sweep region
    // split (ledger copy / the sweep itself / the displaced-release
    // loop + host flags) and the post-harvest tail (the PlanIssue
    // moves) — the drain-tail decode reads which sub-phase pays the
    // cap-specific premium.
    bool const boundary = boundary_breakdown_on();
    std::uint64_t const tb_cq_entry = boundary ? now_ns_local() : 0;
    std::uint64_t tb_rec_done = 0, tb_ledger_done = 0,
                  tb_swp_done = 0, tb_push_done = 0;
    std::size_t tb_displaced = 0;

    // The reconciliation — planned == issued: the log
    // half, the issue log equals the queue op for op (value equality);
    // the sync half, the issued waits equal the derived owed
    // set — one pure function run twice.
    bool ok = slot.issue_log.size() == plan.queue.size();
    if (ok)
    {
        for (std::size_t i = 0; i < slot.issue_log.size(); ++i)
        {
            if (!(slot.issue_log[i] == plan.queue[i]))
            {
                ok = false;
                break;
            }
        }
    }
    if (ok && slot.issued_waits.size() != deriv.owed_set.size())
    {
        ok = false;
    }
    if (ok)
    {
        for (std::size_t i = 0; i < slot.issued_waits.size(); ++i)
        {
            if (slot.issued_waits[i] != deriv.owed_set[i])
            {
                ok = false;
                break;
            }
        }
    }
    if (!ok)
    {
        abort_census(plan, kNoPosition,
            "planned-issued-mismatch: the queue-close reconciliation "
            "found the issued ops or the issued waits differing from the "
            "plan and its derived syncs ");
    }
    if (tracer_ != nullptr)
    {
        tracer_->note_issued(slot.issue_log.size(), true);
    }
    tb_rec_done = boundary ? now_ns_local() : 0;

    // The barriers over ALL lanes — the join and the detection point —
    // DEVICE LANES FIRST, the host lane LAST (a faulted device lane's
    // notify for a host flag never fires, so a host gate would wait
    // forever; the device fault must be delivered at its barrier before
    // the host lane's join can hang). Each
    // device barrier joins the deferral bridge FIRST (the wiring the
    // engine made: deferred enqueues land before the sync, and a
    // bridge fault — a refused deferred step tagged with its op, or a
    // host-lane fault that released a parked gate — is delivered
    // THERE). The census names the bridge fault's exact op when the
    // tag is one of this plan's, else the lane and the LAST OP
    // ENQUEUED ON IT (the lane cannot know the OpRef).
    auto barrier_census = [&](std::string const &lane_text,
                              std::size_t lane_last_op,
                              std::string const &what) {
        std::size_t position = lane_last_op;
        std::string extra;
        if (bridge_->faulted() && bridge_->fault_tag() != 0u)
        {
            // A refused step or a released gate carries its op's ref;
            // a device-fault release carries none (tag 0) — the lane's
            // LAST ENQUEUED OP is the faulting op the census names.
            std::uint64_t const tag = bridge_->fault_tag();
            extra =
                "; the bridge fault's step served op " + render_ref(tag);
            if ((tag >> 32) == plan.plan_seq)
            {
                position = static_cast<std::size_t>(tag & 0xffffffffu);
            }
        }
        abort_census(plan, position,
            std::string("issue-abort: the queue-close barrier on ") +
                lane_text + " delivered a fault: " + what +
                " — the detection point; dependents never run" +
                extra);
    };
    // THE JOIN — ONE shape now, the light close, per lane, all device
    // lanes before the host lane: DRAIN then
    // TERMINAL-EVENT JOIN. The drain (only a TRAILING lane needs it)
    // parks on host progress and record marks, never device
    // completion — the engine's host mints are its own no-op acts —
    // then un-trails the lane (its whole tail is enqueued; the
    // stream's own order carries later direct enqueues). The
    // terminal-event join (the previous engine's batch-granular join,
    // ported) then joins the
    // lane's OWN tail — the event-pool's per-event join (the vendor
    // side's blocking wait, EventPool::join), never a whole-stream
    // sync, so the successor's already-fed prefix never delays the
    // join — and it is the light FAULT PROBE:
    // a device lane's async fault delivers HERE, at the join. Residual
    // ordering seam (named in the bridge's
    // comment): a fault chain whose victim lane's drain precedes its
    // killer's join can park past this close — the exec lanes join
    // first (the common killers); the corner is teardown-released.
    std::uint64_t const t_join0 = now_ns_local();
    // The per-lane park census (diagnosis only): each device lane's
    // drain and terminal-join park in microseconds, absolute clock
    // walls around the join sequence and the post-join sweeps — the
    // boundary drain-tail decode reads WHICH lane the close parks on.
    // (The `boundary` flag is declared at the close_queue entry.)
    std::string lane_parks;
    std::uint64_t tb_first_join = 0, tb_joins_done = 0;
    for (LaneId lane = 0; lane < inv_->lane_count(); ++lane)
    {
        if (lane == host_exec_lane_)
        {
            continue;
        }
        if (boundary)
        {
            if (tb_first_join == 0)
            {
                tb_first_join = now_ns_local();
            }
        }
        bool const trailing = bridge_->lane_trailing(lane);
        bool joined = false;
        std::uint64_t park_drain = 0, park_join = 0;
        if (trailing)
        {
            try
            {
                std::uint64_t const t0 = now_ns_local();
                bridge_->drain_lane(lane);
                close_drain_ns_ +=
                    now_ns_local() - t0;
                park_drain = now_ns_local() - t0;
                if (lane < lane_trailing_cache_.size())
                {
                    lane_trailing_cache_[lane] = false;
                }
            }
            catch (std::exception const &ex)
            {
                barrier_census(inv_->lane_name(lane),
                    slot.last_op_on_lane[lane],
                    ex.what());
            }
            // A trailing lane's tail carries notifies enqueued by
            // its WAITER — cudaLaunchHostFunc callbacks (the
            // host-flag gates) — and an EVENT join can return before
            // the callbacks actually execute (a flag gate racing the
            // registry sweep would hang a later waiter). The tracer's
            // stamps are not in this class: they are the provision's
            // host-clock reads, written at issue before the close —
            // the barriers retire the device work between the
            // positions, and the pair read happens at the harvest
            // after this join. The STREAM SYNC
            // retires the callbacks (the heavy join, correctness
            // first — and the trailing set is the gated set, empty in
            // the steady loop by the protocol fixes).
            try
            {
                std::uint64_t const t0 = now_ns_local();
                lanes_[lane].lane->barrier();
                close_join_ns_ += now_ns_local() - t0;
                park_join = now_ns_local() - t0;
            }
            catch (std::exception const &ex)
            {
                barrier_census(inv_->lane_name(lane),
                    slot.last_op_on_lane[lane],
                    ex.what());
            }
            joined = true; // the sync subsumes the terminal event
        }
        if (!joined &&
            (lane >= slot.last_op_on_lane.size() ||
                slot.last_op_on_lane[lane] == kNoPosition))
        {
            if (boundary)
            {
                lane_parks += std::to_string(lane) + ":tr=" +
                    (trailing ? "1" : "0") + ",dr=0,jn=-1; ";
            }
            continue; // untouched lane: no terminal mint was placed
        }
        if (!joined)
        {
            try
            {
                std::uint64_t const t0 = now_ns_local();
                lanes_[lane].pool->join(slot.terminal_handles[lane]);
                close_join_ns_ += now_ns_local() - t0;
                park_join = now_ns_local() - t0;
            }
            catch (std::exception const &ex)
            {
                barrier_census(inv_->lane_name(lane),
                    slot.last_op_on_lane[lane],
                    ex.what());
            }
        }
        if (boundary)
        {
            lane_parks += std::to_string(lane) + ":tr=" +
                (trailing ? "1" : "0") + ",dr=" +
                std::to_string(park_drain / 1000) + ",jn=" +
                std::to_string(park_join / 1000) + "; ";
            tb_joins_done = now_ns_local();
        }
        if (lane < slot.last_op_on_lane.size() &&
            slot.last_op_on_lane[lane] != kNoPosition)
        {
            lanes_[lane].pool->release(slot.terminal_handles[lane]);
            slot.terminal_handles[lane] = 0;
        }
    }
    try
    {
        std::uint64_t const t0 = now_ns_local();
        host_lane_->barrier();
        close_host_ns_ += now_ns_local() - t0;
    }
    catch (std::exception const &ex)
    {
        barrier_census("the host lane",
            slot.last_op_on_lane[host_exec_lane_],
            ex.what());
    }
    close_total_ns_ += now_ns_local() - t_join0;
    std::uint64_t const tb_host_done = boundary ? now_ns_local() : 0;

    // The bridge must be idle after the per-lane drains (the wiring
    // contract); pending work here means a lane was never drained —
    // loud, never serviced.
    if (!bridge_->idle())
    {
        abort_census(plan, kNoPosition,
            "issue-abort: the deferral bridge carries pending steps "
            "after the close's per-lane drains — an engine defect "
            "");
    }

    // The registry sweep — displacement only — and the
    // retired handles released to their home pools. reg_pre is
    // snapshotted BEFORE the sweep (the label reads honestly: the
    // registry size the close STARTED from, against which the
    // displaced count and the post-sweep size reconcile). The sweep
    // MOVES the displaced entries out in SORTED-REFS order (see the
    // sweep's own comment: the release order changed from the map's
    // iteration order and is inert — the pool is a bitmap), so the
    // close walks ONLY the displaced (the steady close displaces
    // a small fraction of the entries; the old whole-registry ledger
    // copy plus
    // the per-entry find was the dominant close cost); an entry whose
    // op a current fact still names keeps its handle until the run
    // ends, and only a displaced entry's handle returns to its lane
    // family's provision.
    std::size_t const reg_pre = registry_.map.size();
    {
        std::vector<std::pair<OpRef, EventRegistry::Entry>> displaced;
        tb_ledger_done = boundary ? now_ns_local() : 0;
        registry_.sweep(dep_state(), displaced);
        tb_swp_done = boundary ? now_ns_local() : 0;
        for (auto &kv : displaced)
        {
            if (boundary)
            {
                ++tb_displaced;
            }
            try
            {
                (kv.second.pool != nullptr
                        ? kv.second.pool
                        : lanes_[kv.second.owner].pool)
                    ->release(kv.second.handle);
                ++handles_released_;
            }
            catch (std::exception const &ex)
            {
                abort_census(plan, kNoPosition,
                    std::string("issue-abort: the sweep's release of ") +
                        render_ref(kv.first) +
                        " was refused: " + ex.what());
            }
        }
    }

    // The host flags of this plan: every gate has been served at the
    // barriers, so the flags recycle to the host lane's provision
    // (release clears the flag; no waiter exists at the sweep — the
    // flags are counted in neither mint
    // ledger, keeping acquired - released == the live registry size).
    for (auto const &kv : slot.host_flags)
    {
        try
        {
            host_lane_->pool().release(kv.second);
        }
        catch (std::exception const &ex)
        {
            abort_census(plan, kNoPosition,
                std::string("issue-abort: a host flag's release of ") +
                    render_ref(kv.first) + " was refused: " + ex.what());
        }
    }
    std::uint64_t const tb_sweep_done = boundary ? now_ns_local() : 0;

    // The tracer harvest — facts, read only now, never before.
    // The stamps' clock reads are read off their
    // lanes through the stamp rule and composed into the ISSUE
    // TILES: on each lane, the op's interval runs from its own issue
    // read to the next read on that lane (the next op's, or the
    // issue order's tail) — the tile the op owns of the lane's issue
    // order, the duration including the lane's queue wait. The reads
    // recycle their stamps (the provision stands ready for the next
    // plans; the facts are already copied out). An untraced run
    // placed nothing: the walk reads no stamp and the harvest keeps
    // its no-op.
    if (tracer_ != nullptr)
    {
        std::vector<StampFact> facts(plan.queue.size());
        std::vector<StampRef> pending(
            inv_->lane_count(), StampRef{}); // per lane: the open tile
        for (std::size_t p = 0; p < plan.queue.size(); ++p)
        {
            StampRef const &sr = slot.stamp_refs[p];
            if (!sr.placed)
            {
                continue; // LINK/UNLINK, or an untraced issue walk
            }
            ExecLane *lane_ptr = lanes_[sr.lane].lane;
            StampRef &open = pending[sr.lane];
            if (open.placed)
            {
                // The open tile closes at THIS op's issue read.
                facts[open.position] = lane_ptr->stamp_read(
                    open.read, sr.read);
                lane_ptr->stamp_release(open.read);
            }
            open = sr;
        }
        for (LaneId lane = 0; lane < inv_->lane_count(); ++lane)
        {
            StampRef &open = pending[lane];
            if (!open.placed)
            {
                continue;
            }
            ExecLane *lane_ptr = lanes_[lane].lane;
            if (slot.tail_placed[lane] != 0)
            {
                // The plan's last tile on the lane closes at the
                // issue order's tail read.
                facts[open.position] =
                    lane_ptr->stamp_read(open.read,
                        slot.tail_stamps[lane]);
                lane_ptr->stamp_release(slot.tail_stamps[lane]);
            }
            else
            {
                // No tail (an abort mid-issue left the plan
                // unclosed): the open tile has no end — the run is
                // aborting anyway; the stamp releases without a
                // fact.
                facts[open.position] = StampFact{};
            }
            lane_ptr->stamp_release(open.read);
            open = StampRef{};
        }
        tracer_->harvest(plan.plan_seq, facts);
    }
    std::uint64_t const tb_harvest_done = boundary ? now_ns_local() : 0;

    // The slot's buffers retire with the slot itself (the close drops
    // it): the operand-address arrays and the hook arguments were
    // alive until these barriers joined them.

    // The witness. (Retention policy: the run-order history keeps the
    // SCALAR verdict per plan — the
    // meters fold reconcile_ok over it and nothing else reads the
    // history's payload — while the FULL payload (issue_log +
    // issued_waits) lives ONCE, in last_issue(): the engine's
    // note_batch walk reads it there and the executor suite checks it
    // there. The earlier form deep-copied both vectors into the
    // history at every close (a large log plus one heap allocation per
    // position's wait vector — a measurable per-close cost); the
    // retention now
    // moves the payload once and frees the previous plan's.
    PlanIssue issue;
    issue.plan_seq = plan.plan_seq;
    issue.issue_log = std::move(slot.issue_log);
    issue.issued_waits = std::move(slot.issued_waits);
    issue.reconcile_ok = true;
    issue.mints_enqueued =
        static_cast<std::size_t>(
            std::count(deriv.mints.begin(), deriv.mints.end(), true)) +
        deriv.late_records.size();
    issue.handles_released =
        static_cast<std::size_t>(handles_released_ - released_at_entry);
    PlanIssue retained;
    retained.plan_seq = issue.plan_seq;
    retained.reconcile_ok = issue.reconcile_ok;
    retained.mints_enqueued = issue.mints_enqueued;
    retained.handles_released = issue.handles_released;
    issues_.push_back(std::move(retained));
    last_full_ = std::move(issue);
    have_last_ = true;
    tb_push_done = boundary ? now_ns_local() : 0;
    if (boundary)
    {
        std::fprintf(stderr,
            "[nnhaul-boundary-close] seq=%llu t_entry=%llu "
            "t_rec=%llu t_join0=%llu "
            "t_firstjoin=%llu t_joinsdone=%llu t_host=%llu "
            "t_ledger=%llu t_swp=%llu t_sweep=%llu t_harvest=%llu "
            "t_push=%llu reg_pre=%zu displaced=%zu lanes=[%s]\n",
            static_cast<unsigned long long>(plan.plan_seq),
            static_cast<unsigned long long>(tb_cq_entry),
            static_cast<unsigned long long>(tb_rec_done),
            static_cast<unsigned long long>(t_join0),
            static_cast<unsigned long long>(tb_first_join),
            static_cast<unsigned long long>(tb_joins_done),
            static_cast<unsigned long long>(tb_host_done),
            static_cast<unsigned long long>(tb_ledger_done),
            static_cast<unsigned long long>(tb_swp_done),
            static_cast<unsigned long long>(tb_sweep_done),
            static_cast<unsigned long long>(tb_harvest_done),
            static_cast<unsigned long long>(tb_push_done),
            reg_pre,
            tb_displaced,
            lane_parks.c_str());
    }
} // close_queue

} // namespace nnhaul
