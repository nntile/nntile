/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * execute-mode issue loop of the chunk-graph executor over the run's
 * lanes: the boundary check, the one dependency pass's mint acquisitions,
 * the literal per-op issue, and the queue close.
 *
 * @file src/plan_executor.hh
 * The execute-mode issue loop. execute(plan) runs line by line:
 * (0) the boundary check, applied.equals(P.entry) over ALL nodes'
 * segments, else boundary-mismatch BEFORE any op issues; (1) the
 * derivation pass, Derivation.run(P) — the one dependency pass whose
 * run-scoped state carries every earlier plan's facts, INCLUDING the
 * still-running ones (enqueue-before-derive is the ENGINE's calling
 * discipline: plan k+1's derivation follows plan k's last ENQUEUE, not
 * its barriers; this executor asserts plan_seq continuity loudly) —
 * then the mint acquisitions from the lane-family pools for every
 * minted op INCLUDING the late records, each registered in the one
 * EventRegistry (never two on one op), the derivation-time enqueues
 * (late records, host-flag notifies) riding the single submission path
 * so a bridge still trailing an earlier plan keeps the one issue
 * order; (2) the loop over P.queue in order, per op: derived waits
 * first (resolved through the registry, absent -> registry-miss,
 * enqueued on the op's own lane BEFORE its work), the tracer's ONE
 * issue-position stamp (the tile closes at the next issue read or the
 * plan's tail stamp; no fire notify exists — completion facts ride the
 * queue-close barriers), the dispatch (LINK/UNLINK a binding-table
 * transition only; COPY one occupied-span copy on chan(src,dst) — the
 * byte count derived from the tile table; TILE_OP one launch on
 * exec(node)), the mint-if-owed immediately after the work, the applied
 * table's same fold transition with the rows re-evaluated abort-only
 * against the pre-op state, and the issue-log append — ISSUE ends here,
 * its state RETAINED IN A SLOT (depth-2: two un-joined plans stand at
 * once, every in-flight buffer living in its own slot); (3) close_queue
 * (the JOIN act): the planned == issued reconciliation (the log half op
 * for op, the sync half issued waits == the derived owed set — one pure
 * function run twice), barriers on EVERY lane DEVICE LANES FIRST and
 * the host lane last (a faulted device lane's barrier delivers the
 * fault before the host lane's joins can hang on a gate whose notify
 * never fired), the registry sweep with the retired handles released to
 * their pools (displacement only), and the tracer harvest (facts, read
 * only now).
 *
 * The wait matrix, and the one DECISION this file owns: a HOST-lane
 * consumer gates on a HOST flag acquired at the derivation pass, the
 * flag SET by a producer-lane notify enqueued immediately after the
 * mint's record (never a poll); a DEVICE-lane consumer of a HOST-family
 * mint is realized by DEFERRED ENQUEUE, realized BACKEND-SIDE by
 * src/host_worker.hh's DeferralBridge: from the first such wait of a
 * plan on, EVERY lane enqueue of the plan — waits, stamps, device work,
 * mints, the host consumer's own gate, the flag-setting notifies, all
 * of them, ONE issue order — appends to the bridge's single FIFO, whose
 * waiter thread executes the steps in that order, parking each host-gate
 * step until the mint's flag is host-visible (through the host lane's
 * own cv machinery — fault- and teardown-aware, never a spin). The
 * single order is the soundness argument: a host consumer's gate can
 * never overtake the deferred notify that sets its flag, and the
 * mint-capture guarantee (every record call precedes its wait call)
 * holds per lane by construction. The NOT-chosen mechanism, the gate
 * host function of cuda_backend.hh's wait_host_flag (a cudaLaunchHostFunc
 * BLOCKING on the worker's flag), is empirically a driver wedge on this
 * stack (reproducible: with the gate in flight, later cudaLaunchKernel
 * calls on the engine thread stall inside libcuda while the flag's own
 * setter chain — a notify host function — cannot execute): a blocking
 * host function is not a safe v1 default under pipelined submission.
 * The bridge joins through the LANES' BARRIERS — the engine wires it via
 * set_deferral_bridge and every CUDA lane's barrier() drains the FIFO
 * before its sync — so the engine plane's only blocking acts remain
 * lane.barrier(); a refused deferred step, or a HOST-lane fault
 * releasing a parked gate (a faulted host lane never mints the flags
 * behind it), surfaces at that join carrying the step's tag — the
 * executor encodes the served op's OpRef — and the census names the
 * exact op. The close join is SYNC-FIRST across the watched device
 * lanes (the bridge's join_barrier, run through each lane's barrier()):
 * every lane syncs before the drain, a faulted sync RELEASES every
 * parked gate into the fault path (the host lane faulted from outside —
 * the worker's gates lift, its remaining items are skipped, dependents
 * never run), and the final barrier drains then re-syncs every lane,
 * joining the deferred tail. NO PARK CAN OUTLIVE THE SYNC OF ITS
 * REGISTERED LANE — the device-fault drain-park is closed, not
 * documented away.
 *
 * Zero event-state queries anywhere: the host never waits on, polls, or
 * reads an event; host knowledge comes only from notifications and the
 * queue-close barriers. Submission is non-blocking on the engine thread:
 * nothing in the loop waits on device work — the ONLY engine-thread
 * blocking acts are the lane barriers. Error propagation: an enqueue
 * refusal aborts immediately with a census naming the op (kind,
 * position, OpRef), dependents never run — nothing further is enqueued
 * on any lane, a refusal inside a deferred step surfaces at the barrier
 * join with the census its tag names; the queue-close barrier is THE
 * detection point, its fault wrapped with the census (the lane knows its
 * device, the executor knows the op). A fault is never serviced.
 *
 * Ownership, decided and pinned for the engine piece: the executor
 * OWNS the run-scoped Derivation (with its DependencyState), the one
 * EventRegistry, and the applied BindingTable — the planner and the
 * executor share run-scoped state through the executor's exposed
 * boundary products (the applied table for the initial load and the
 * boundary read, the dependency state for the timing retire seam, the
 * last PlanDerivation's owed sets for simulate over the same
 * derivation). The lanes, the pools, the bridge, and the address
 * surfaces are BORROWED for the executor's life — the engine builds
 * them from the backends and the executor stays vendor-free: only
 * src/backend.hh's interface plus host_worker.hh (the host lane's own
 * type and the DeferralBridge — backend-side, vendor-free) appear here.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include "backend.hh"       // ExecLane, TransferChannel, EventPool
#include "binding_table.hh" // the applied table
#include "dependency.hh"    // Derivation, EventRegistry
#include "host_worker.hh"   // HostWorkerLane + HostEventPool
                           // + DeferralBridge (deferred enqueue,
                           // backend-side, vendor-free)
#include "plan.hh"          // PlanOp, Plan, Census
#include "plan_trace.hh"    // PlanTrace: stamps + fire ring

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map> // the copy-run break census's gap histograms
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nnhaul
{

// ---- the engine-bound seam (vendor-free; the engine binds the
//      backends' own surfaces as data) ----

// One node's chunk-address surface (addr : ChunkId -> chunk
// address, valid and stable for the run's life). The engine binds each
// device node's lane arena address (CudaExecLane::addr) and the host
// node's pinned pool address (HostPool::addr); a COPY's endpoints
// resolve through these, never through a vendor type here.
using ChunkAddresser = std::function<void *(ChunkId)>;
// The chunk's CONTIGUOUS EXTENT in bytes from its base — the span of
// the ONE allocation the chunk lives in (a copy may not cross an
// allocation boundary on the device: cudaMemcpyAsync rejects ranges
// spanning two allocations even when they are address-adjacent). The
// segmented pools report the chunk's remaining span to its SEGMENT's
// end — the COPY-run merge judges each extension against the RUN HEAD's
// span, so a run fuses exactly as far as the head's own allocation
// reaches.
using ChunkContig = std::function<std::size_t(ChunkId)>;

// ---- the execution-time census error  ----

// The loud execution-time failure: the census travels beside the plan,
// never in it, and the run stops — never serviced, never retried. The
// fault names are boundary-mismatch, planned-issued-mismatch,
// issue-abort, and registry-miss, with the finding, the position, the
// offending op verbatim, the fold and applied renders, the entry state,
// the ops prefix, and the issue log prefix (execution time's own extra
// facts).
struct ExecutorCensusError : std::runtime_error
{
    Census census;

    ExecutorCensusError(std::string const &reason, Census payload) :
        std::runtime_error(reason), census(std::move(payload))
    {
    }
};

// ---- the executor  ----

class PlanExecutor
{
  public:
    // The run's bindings, all borrowed for the executor's life: `inv`
    // the static inventory (the lane set is its truth); `exec_lanes`
    // one execution lane per node indexed by NodeId, the host entry
    // being `host_lane` (the host node's own lane type — its pool IS
    // the host family's EventPool, and its flag array is the deferral
    // bridge's seam); `channels` one transfer channel per connected
    // ordered pair, DENSE IN CHANNEL-LANE ID ORDER (the inventory
    // numbers chan lanes in ascending (a, b) order of the connected
    // pairs — the engine walks the same order); `device_pools` one
    // EventPool per device node indexed by NodeId; `addr` one
    // ChunkAddresser per node; `bridge` the backend-side DeferralBridge
    // the engine built over the host lane AND WIRED INTO EVERY DEVICE
    // LANE (set_deferral_barrier — the lanes' barrier() drains it
    // first; the executor censuses an unwired bridge loudly at the
    // close); `tiles` the run's tile table — the registered payload
    // bytes every COPY's occupied span derives from
    // (tile_chunk_span(nbytes(tile), index)), the one byte-count source
    // the dispatch reads; `tracer` optional and possibly disarmed (an
    // untraced run places none of the notifies). Every defect of the
    // binding is a loud construction error, never serviced.
    PlanExecutor(NodeInventory const &inv,
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
        PlanTrace *tracer = nullptr);

    ~PlanExecutor() = default;

    PlanExecutor(PlanExecutor const &) = delete;
    PlanExecutor &operator=(PlanExecutor const &) = delete;
    PlanExecutor(PlanExecutor &&) = delete;
    PlanExecutor &operator=(PlanExecutor &&) = delete;

    // ---- the boundary products (the planner's and the engine's run-
    //      scoped surfaces; the executor owns all three) ----

    // The APPLIED binding table — the memory state the issue loop
    // advances by the fold's own transitions. The engine performs the
    // run's initial load through the same transitions at init; the
    // boundary check reads it at every issue start; the successor's
    // chain input is the planner's PREDICTED exit fold, and this
    // table's snapshot is the applied exit the census renders beside
    // it.
    BindingTable &applied();
    BindingTable const &applied() const;

    // The carried dependency state — the sweep's input
    // and the run's trailing facts; the engine derives the timing
    // pass's live refs from it (the retire seam of src/timing.hh).
    DependencyState const &dep_state() const;

    // Grow node n's slot range to new_cardinality in the
    // executor's world: the applied binding table (new chunks free)
    // and the carried dependency rows (dense re-base). The engine
    // commits pool memory first; this follows at the same seam.
    void apply_growth(NodeId n, std::size_t new_cardinality);


    // The one registry. Non-const on purpose: a test wipes an entry
    // between the derivation and the issue — a documented TEST seam;
    // the engine itself never edits a mint.
    EventRegistry const &registry() const;
    EventRegistry &registry();

    // The last plan's derivation products — the owed sets the engine
    // hands the timing pass (simulate consumes the same folded sets the
    // executor issued), the mint flags, the late records. The SLOT form
    // names an un-joined plan's own products (the engine's simulate
    // runs at the JOIN).
    PlanDerivation const &last_derivation() const;
    PlanDerivation const &slot_derivation(std::size_t slot) const;

    // ---- the planned == issued witness, queryable per
    //      plan in run order ----

    struct PlanIssue
    {
        std::uint64_t plan_seq = 0;
        std::vector<PlanOp> issue_log; // the issued ops, verbatim, in
                                       // issue order — one entry per
                                       // issued op, nothing else
        std::vector<std::vector<OpRef>> issued_waits; // per position,
                                                      // the owed refs
                                                      // whose waits
                                                      // were actually
                                                      // enqueued
        bool reconcile_ok = false; // the queue-close verdict
        std::size_t mints_enqueued = 0; // this plan's mint records:
                                        // own mints + late records
        std::size_t handles_released = 0; // at this plan's sweep
    };

    PlanIssue const &last_issue() const;
    std::vector<PlanIssue> const &issues() const; // run order

    // ---- the pass (execute(plan P)), SPLIT into its two acts
    //      (depth-2 pipelining — two un-retired batches; the join act IS
    //      the older plan's queue-close barriers): ISSUE enqueues the
    //      whole queue and returns; CLOSE joins — reconcile, barriers,
    //      sweep, harvest. The fused execute() below remains for the
    //      conformance suites. Throws ExecutorCensusError on every census
    //      (boundary-mismatch, registry-miss, issue-abort,
    //      planned-issued-mismatch) and on the executor's own
    //      discipline assertion (a plan_seq discontinuity — the
    //      enqueue-before-derive calling discipline is the engine's to
    //      keep and this executor's to demand: plan k+1's derivation
    //      runs only after plan k's LAST enqueue — transitions included
    //      — not after its barriers). Nothing is serviced on a fault:
    //      the run stops with the plan's acquired-but-unreleased state
    //      standing (a fresh run re-inits).

    // ISSUE: the boundary check, the derivation pass with its mint
    // registrations, and the literal issue loop — every enqueue
    // through the single submission path, so a plan issued while an
    // earlier plan still trails the deferral bridge keeps the one
    // issue order (the late records and host-flag notifies of THIS
    // pass ride submit_ too — the successor-while-predecessor-runs
    // case the machinery exists for). Returns the retained SLOT the
    // close will join; at most two slots stand un-joined (the
    // engine's depth-2 discipline — the executor itself demands only
    // the run-queue order).
    std::size_t issue(Plan const &plan);

    // CLOSE: the slot's queue close — the planned == issued
    // reconciliation, the barriers over ALL lanes (device lanes
    // first, the host lane last — the deadlock seam), the registry
    // sweep with the retired handles released, and the tracer
    // harvest. Joins the slot's whole enqueued tail — and, through
    // the bridge-draining barriers, every step any LATER-issued plan
    // appended before this close began (sound: the later plan's issue
    // completed first, in one order). The slot retires; its full
    // PlanIssue witness is last_issue().
    void close(Plan const &plan, std::size_t slot);

    // The fused form (issue + close, the conformance suites' shape
    // and the engine's boundary acts): the synchronous executor of a
    // single plan.
    void execute(Plan const &plan)
    {
        std::size_t const slot = issue(plan);
        close(plan, slot);
    }

    // The deferral mechanism's state — the engine's registration-order
    // discipline, PER LANE: while a LANE trails the bridge (its queue
    // holds steps; its waiter launches), the engine's own lane-table
    // registrations for that lane must ride the SAME queue or they
    // would race the waiter's concurrent launches.
    bool deferral_trailing(LaneId lane) const
    {
        return bridge_->lane_trailing(lane);
    }

    // Cumulative counters — the no-op-issued witness of the boundary
    // abort and the mint accounting (acquired once, released at the
    // sweep; the outstanding difference is exactly the live registry
    // plus nothing — the host flags acquire and release within one
    // plan).
    std::uint64_t ops_issued() const;
    std::uint64_t mints_acquired_total() const;
    std::uint64_t handles_released_total() const;
    std::uint64_t plans_executed() const;

    NodeInventory const &inventory() const { return *inv_; }

    // The diagnosis line (NNHAUL_GATE_TRACE=1): the run's histogram
    // of device-waits-host gates by the WAITED host op's task symbol —
    // which mint family parks the bridge. Diagnosis only, never
    // control flow.
    std::unordered_map<std::uint64_t, std::uint64_t> const &
    gate_waited_tasks() const
    {
        return gate_waited_tasks_;
    }
    std::uint64_t gates_total() const { return gates_total_; }
    std::unordered_map<std::uint64_t, std::uint64_t> const &
    gate_consumers() const
    {
        return gate_consumers_;
    }

    // The close's split meters (the performance wave's diagnosis
    // line): the terminal-event joins, the per-lane drains, the host
    // barrier, and the close's whole span — GPU-bound versus
    // deferral-bound in four numbers.
    std::uint64_t close_join_ns() const { return close_join_ns_; }
    std::uint64_t close_drain_ns() const { return close_drain_ns_; }
    std::uint64_t close_host_ns() const { return close_host_ns_; }
    std::uint64_t close_total_ns() const { return close_total_ns_; }

    // ---- the documented TEST seam: invoked on the engine thread
    //      AFTER the derivation pass and its mint registrations and
    //      BEFORE the first op issues — the one place a test can
    // one place a test can
    //      break an engine rule (wipe a registry entry) between
    //      derive and issue. The engine never sets it.
    void set_post_derivation_hook(
        std::function<void(Plan const &, PlanDerivation const &)> hook)
    {
        post_derivation_hook_ = std::move(hook);
    }

  private:
    // One lane's bindings, resolved once at construction: the lane
    // every call goes to, the transfer channel when the lane is one
    // (the copy lives there), and the lane family's event
    // provision — the mint handles' home.
    struct LaneSlot
    {
        ExecLane *lane = nullptr;
        TransferChannel *chan = nullptr; // channel lanes only
        EventPool *pool = nullptr;       // the mint handles' home family
    };

    // The census of a fault at op `position` (kNoPosition = entry /
    // plan level): the finding rendered with the position and the
    // OpRef, the fold and applied renders, the entry, the prefixes,
    // and the inputs' relevant facts.
    static constexpr std::size_t kNoPosition =
        static_cast<std::size_t>(-1);

    [[noreturn]] void abort_census(
        Plan const &plan, std::size_t position, std::string const &finding) const;

    // The op rendered for a census line: kind, queue position, OpRef.
    static std::string op_line(Plan const &plan, std::size_t position);

    // The light row re-evaluation (abort-only, free redundancy): the
    // memory-legality rows the APPLIED table and the static inventory
    // can answer — ids, identity/linkage, topology, flags,
    // distinctness — evaluated against the PRE-op applied state (an
    // op's preconditions read the fold of all earlier ops; rows before
    // effects). The tile-table and executability rows are the
    // checker's inputs, not the executor's, and stay out. Returns the
    // violated row's fault name verbatim, empty when every row holds.
    std::string light_rows(PlanOp const &op) const;

    // The fold's transitions over applied_ — the SAME calls the
    // checker and the planner's state read make (binding_table.hh's
    // contract).
    void apply_transition(PlanOp const &op);

    // The wait matrix: one wait per folded cross-lane owed ref, on the
    // consumer's own lane, before its work — every one of them through
    // the single submission path. A HOST consumer waits a host flag this
    // plan acquired for the mint (the producer lane's notify set it —
    // enqueued at the mint's enqueue time, never a poll); a DEVICE
    // consumer of a HOST-family mint parks the deferral bridge (the file
    // comment's decision); otherwise the handle verbatim. `ref` names the
    // consumed op for the census wrap and the bridge's tag.
    void issue_wait(
        Plan const &plan, std::size_t position, LaneId consumer, OpRef ref);

    // One op's device work. The
    // operand-address array is kept alive until the queue-close
    // barrier retires it (the host lane's capture contract).
    void issue_work(Plan const &plan, std::size_t position, PlanOp const &op);


    // The mint's host-flag companion: acquire (derivation pass) and
    // enqueue (at the mint's enqueue position) of the host flag a
    // host-lane consumer gates on, plus the hook that sets it.
    EventHandle acquire_host_flag();
    void enqueue_host_flag_notify(LaneId producer_lane, OpRef producer);

    // ---- the submission path: ONE issue order ----

    // The engine thread's enqueue: direct while this lane trails
    // nothing (every backend call records and returns), and handed to
    // the backend-side DeferralBridge in strict issue order once the
    // first device-waits-host wait parked it — whatever the lane,
    // whatever the call, waits and the host consumer's own gate
    // included. `tag` is the served op's OpRef (the producer's ref for
    // a flag-setting notify): the bridge's fault census names it back.
    // The twin-mint enqueue: the FIFO Mint + the record twin on the
    // host lane, the plain record on every device lane.
    void mint_with_twin_(LaneId lane, EventHandle handle, OpRef ref);

    void submit_(LaneId lane, OpRef tag, std::function<void()> fn,
        bool closes_copy_run = true);
    void submit_host_gate_(LaneId lane, OpRef tag, EventHandle e,
        LaneId producer);
    // ^ per-lane: a gate parks only the CONSUMER's lane; the minting
    // lane its flag-setting notify rides is the census's key.

    // ---- the COPY-run realization ----
    //
    // The backend realizes consecutive same-lane COPY ops whose src
    // AND dst spans are ADDRESS-CONTIGUOUS as ONE driver call (the
    // chan lane's copy_run): the plan's queue, the issue log, the
    // per-op stamps and trace events, and every op's byte accounting
    // stay untouched — only the stream submission merges.
    // A lane's OPEN run absorbs, in queue order: the member COPYs'
    // spans (address-checked), the member ops' own cudaStreamWaitEvent
    // steps (submitted inline — the merged bytes ride behind EVERY
    // member's waits, a superset of the unmerged gating that adds only
    // backward edges, never a cycle: waits name strictly earlier ops
    // under enqueue-before-derive), the tracer's stamps (a
    // host clock read, zero CUDA API), and the members' mint records
    // (DEFERRED to the run's close, in issue order, still after the
    // merged bytes). Everything else that steps the lane — a notify,
    // a wait resolving a DEFERRED mint of the run, a
    // non-contiguous successor COPY, the plan's issue end — CLOSES the
    // run first (its bytes precede whatever orders them). The
    // mint-capture rule is preserved exactly: a consumer resolving a
    // member's mint flushes the owner's run BEFORE the wait's call, so
    // the record's call still strictly precedes (issue_wait's hook).
    void flush_copy_run_(LaneId lane);
    void flush_all_copy_runs_();

    // The record-step stamp, ONE helper for every stamp site:
    // writes the dense epoch row (the elision fast path's source) and
    // the registry entry's rec_epoch (the full path's source) from
    // the same counter tick, so the two views are identical by
    // construction.
    void stamp_rec_(OpRef ref, LaneId owner);

    // One lane's open run (indexed by LaneId; only chan lanes ever
    // open one — TILE_OPs ride exec lanes and COPYs ride chan lanes).
    struct CopyRun
    {
        bool open = false;
        OpRef tag = 0; // the leader's ref (the flush step's census tag)
        void const *src = nullptr; // the run's base span
        void *dst = nullptr;
        // The run's OPENING chunks, both endpoints: the allocation
        // bound is judged from the run's HEAD, never the member being
        // appended — contig_ counts the span to the segment's end from
        // a chunk, and only the head's count covers the whole run.
        ChunkId src_chunk = 0;
        ChunkId dst_chunk = 0;
        std::size_t nbytes = 0; // the summed occupied spans
        std::size_t ops = 0;    // planned COPY ops realized by the run
        std::vector<std::pair<OpRef, EventHandle>> mints; // deferred
                                                          // records, in
                                                          // issue order
    };
    std::vector<CopyRun> copy_runs_; // indexed by LaneId

    // diagnosis counters (NNHAUL_COPY_RUNS prints them per plan):
    // planned COPY ops realized in runs and the driver calls they
    // saved. Never control flow.
    std::uint64_t copy_run_ops_ = 0;
    std::uint64_t copy_run_calls_saved_ = 0;

    // The per-COPY executor-path diagnosis (NNHAUL_COPY_BREAKDOWN,
    // diagnosis only, never control flow): the wait center split into
    // its resolution walk and its submit (the enqueue incl. the inline
    // cudaStreamWaitEvent), the flush's memcpy submit apart from the
    // deferred member-mint records, the COPY stamps, and the rows —
    // the (i)-(iv) split of the issue-loop COPY cost. Same-lane counts
    // the issue_wait calls whose mint owner IS the consumer's own lane
    // (the same-lane elision should already make this zero — the census
    // proves it per plan).
    struct CopyDiag
    {
        std::uint64_t copy_ops = 0;
        std::uint64_t wait_n = 0;    // COPY-consumer issue_wait calls
        std::uint64_t wait_ns = 0;   // their whole walk
        std::uint64_t wres_ns = 0;   // registry resolve + flush-hook scan
        std::uint64_t wsub_ns = 0;   // the wait enqueue (incl. driver)
        std::uint64_t samelane = 0;  // owner == consumer
        std::uint64_t flushhook = 0; // waits that flushed an open run
        std::uint64_t flush_n = 0;   // flush_copy_run_ with bytes
        std::uint64_t flprep_ns = 0; // the run bookkeeping + lambda
        std::uint64_t flcopy_ns = 0; // the copy_run submit (memcpy)
        std::uint64_t mrec_n = 0;    // deferred member records at flush
        std::uint64_t mrec_ns = 0;   // their submit loop
        std::uint64_t stamp_n = 0;   // COPY stamps placed
        std::uint64_t stamp_ns = 0;
        std::uint64_t rows_ns = 0;   // light_rows over COPY ops
        // The TILE_OP-consumer wait split (diagnosis only): the same
        // res/sub centers as the COPY split above, for the waits
        // TILE_OPs owe — often the fold's cross-lane copy edges, the
        // largest component of the issue pass.
        std::uint64_t tilo_wait_n = 0;
        std::uint64_t tilo_wait_ns = 0;
        std::uint64_t tilo_wres_ns = 0;
    };
    CopyDiag copy_diag_;
    bool copy_diag_on_ = false; // NNHAUL_COPY_BREAKDOWN, read once

    // THE TRANSITIVE STREAM-WAIT ELISION. A per-(consumer, owner)
    // high-water of waited record epochs makes a new wait for an epoch
    // <= the high-water a provable no-op, provided the consumer's
    // stream already holds a wait for an event whose record is
    // stream-ordered at least as late as this one's.
    //
    // Soundness: every mint-record submit_ call stamps its registry
    // entry with a globally increasing engine-thread epoch
    // (next_rec_epoch_; late records at derive, direct loop mints, and
    // a COPY run's deferred members at their flush — every site that
    // submits a record step). The epoch order does NOT strictly certify
    // RECORD order — the inversion is the direct-mint-extending-open-
    // run case: a host-flag COPY absorbed into an open run mints DIRECT
    // (its stamp precedes its submit_), and that submit_ FLUSHES the
    // run, stamping the earlier members AFTER it — so the smaller-epoch
    // direct record submits after the larger-epoch member records. The
    // certificate that holds is over BYTES: in the inversion the direct
    // mint's op is a run member in bytes — its span rides the run's ONE
    // memcpy, which precedes EVERY record the flush submits (members
    // and direct alike) on the same in-order stream; in every other
    // shape the stamps precede the submits in matching order and record
    // order follows epoch order outright. Either way, on one owner lane
    // epoch(r) < epoch(r') certifies r's BYTES are stream-ordered
    // before record(r'). A consumer lane's per-(consumer,owner)
    // high-water of WAITED epochs then makes a new wait for epoch
    // e <= hw a provable no-op: the consumer's stream already holds
    // (or holds in FIFO position, if parked) a wait for event(r_hw),
    // which cannot fire before the earlier epoch's bytes have landed —
    // the elided edge holds a fortiori.
    // Unlike a plain ref comparison this also covers the hazard of a
    // pre-existing mint (high ref, record submitted in an EARLIER plan)
    // waited before a late-recorded LOWER ref (submitted at this plan's
    // derive): the epochs order correctly where the refs do not. An
    // unstamped entry (epoch 0 — a record not yet submitted) is never
    // elided against, and by the mint-capture rule every waited ref's
    // record submit_ call strictly precedes its wait's call, so the
    // stamp is always present at the check (the COPY-run flush hook
    // runs first and stamps run-deferred members). Fault caveat: if the
    // high-water wait PARKS on a record that a lane fault prevents, the
    // fault aborts the run anyway.
    // The high-water is PER PLAN, reset at issue start. Host-lane
    // CONSUMERS are untouched (they gate on host flags, not stream
    // waits).
    std::uint64_t next_rec_epoch_ = 1; // 0 reserved: not yet submitted
    std::vector<std::uint64_t> wait_hw_; // lane_count^2, reset per plan
    bool elision_on_ = true; // NNHAUL_NO_COPY_ELISION, read once
    std::uint64_t k9_elided_ = 0;       // per-plan census (elided waits;
                                         // fast-elided = dense-row hits)
    std::uint64_t k9_waited_ = 0;       // per-plan census (enqueued waits)
    std::uint64_t k9_fast_elided_ = 0;

    // The segment census (diagnosis only, rides the k5/k9 knob):
    // mint-record submits (every stamp_rec_ call: late records, loop
    // mints, and a COPY run's deferred members at their flush — the
    // cudaEventRecord-class call count) and the stream-wait-class
    // enqueues issue_wait performs (the device paths; a parked
    // recorded-wait enqueues the same wait at its mark, the host
    // consumer's flag gate is a host cv wait, not counted).
    std::uint64_t r2_rec_submits_ = 0;
    std::uint64_t r2_stream_waits_ = 0;

    // The dense EPOCH ROWS — the elision check's fast path (the same
    // dense-row shape as the derive facts: OpRef packs (plan_seq,
    // position), plans issue ascending, so the ref IS a dense address).
    // One row per plan seq, one (epoch, owner) cell per queue position,
    // written at the same stamp sites as Entry::rec_epoch (both stores
    // ride stamp_rec_, so the two views cannot diverge). A wait whose
    // ref hits a STAMPED cell (epoch != 0) can be elided or high-water-
    // updated WITHOUT the registry hash resolve: a stamped cell
    // certifies the record submit_ call happened exactly as the
    // registry entry does. A MISS (gap seq, never-stamped position, or
    // a not-yet-flushed COPY-run member) falls to the full path —
    // resolve (the registry-miss guard), the COPY-run flush hook, then
    // the registry-epoch check — so the abort semantics and the
    // deferral handling are byte-identical to the registry-only path.
    // Rows live run-lifetime (late records stamp earlier seqs' cells),
    // the same scope as the registry itself.
    struct RecEpochRow
    {
        std::vector<std::uint64_t> epoch; // 0 = never stamped
        std::vector<LaneId> owner;        // valid iff epoch != 0
    };
    std::vector<RecEpochRow> rec_epoch_rows_; // indexed by plan_seq

    // The break census (NNHAUL_COPY_RUNS, diagnosis only): for
    // every COPY, the channel's direction — 0 d2h (device src, host
    // dst), 1 h2d (host src, device dst), 2 d2d — and, for the ones
    // that failed to extend the lane's OPEN run, which endpoint gapped
    // (src alone / dst alone / both) plus each gapped endpoint's
    // signed distance in class blocks (a histogram; unaligned
    // distances — a tail span's end — counted apart). A fresh open
    // after a closer (run.open == false at the dispatch) counts as a
    // reopen, not a break: closers are order facts, not address gaps.
    // Never control flow; counted only when the diagnosis is armed.
    struct CopyRunDiag
    {
        std::uint64_t ops[3] = {0, 0, 0};    // COPY ops by direction
        std::uint64_t merged[3] = {0, 0, 0}; // joined an open run
        std::uint64_t broke_src[3] = {0, 0, 0};
        std::uint64_t broke_dst[3] = {0, 0, 0};
        std::uint64_t broke_both[3] = {0, 0, 0};
        std::uint64_t src_al[3] = {0, 0, 0}; // src gap class-aligned
        std::uint64_t dst_al[3] = {0, 0, 0}; // dst gap class-aligned
        std::uint64_t reopened[3] = {0, 0, 0};
        std::uint64_t src_unaligned = 0; // a tail-span end (expected)
        std::uint64_t dst_unaligned = 0;
        std::map<long long, std::uint64_t> src_gaps; // class multiples
        std::map<long long, std::uint64_t> dst_gaps;
    };
    CopyRunDiag copy_run_diag_;
    bool copy_run_diag_on_ = false; // NNHAUL_COPY_RUNS, read once

    NodeInventory const *inv_;
    HostWorkerLane *host_lane_ = nullptr;
    DeferralBridge *bridge_ = nullptr;
    PlanTrace *tracer_ = nullptr;
    TileTable const *tiles_ = nullptr; // the COPY span's byte source
                                       // (borrowed, the run's own
                                       // table)

    // The host-mint twins: a HOST-lane
    // op's mint is a DEVICE-pool event recorded by the host worker AT
    // ITS FIFO POSITION (`host_mint_record`, engine-bound — a
    // cudaEventRecord on the dedicated host-mint stream), so a device
    // consumer's wait is a plain STREAM WAIT once the record executed
    // (the mark). The deferral mechanism covers the not-yet-recorded
    // window (the recorded-wait), fault-released both ways.
    EventPool *host_mint_pool_ = nullptr;
    std::function<void(EventHandle)> host_mint_record_;


    std::vector<LaneSlot> lanes_; // indexed by LaneId, lane_count()
    LaneId host_exec_lane_ = kNoLane;
    std::vector<ChunkAddresser> addr_; // indexed by NodeId
    std::vector<ChunkContig> contig_;  // per node

    // The run-scoped state this executor owns (the ownership decision
    // of the file comment): the one dependency pass with its dependency
    // state, the one registry, the applied table.
    Derivation derivation_;
    EventRegistry registry_;
    BindingTable applied_;

    // The diagnosis histogram (env-gated at construction): the waited
    // host op's task per gate, and the run's gate count.
    bool gate_trace_ = false;
    std::unordered_map<OpRef, TaskId> host_op_task_;
    std::unordered_map<std::uint64_t, std::uint64_t> gate_waited_tasks_;
    // key: same_plan*100 + consumer kind (Link=0 Unlink=1 Copy=2
    // TileOp=3): who waits the gated host mints.
    std::unordered_map<std::uint64_t, std::uint64_t> gate_consumers_;
    std::uint64_t gates_total_ = 0;
    // The engine-thread TRAILING CACHE: per lane, the executor's own
    // mirror of the bridge's trailing flag — set when THIS thread
    // parks a gate or a recorded-wait on the lane, cleared at the
    // close's drain. Single-writer (the engine thread), so submit_'s
    // hot path never touches the bridge's mutex: a non-trailing lane
    // enqueues DIRECT, exactly the pre-deferral cost (every enqueue
    // passing one global mutex was slow).
    std::vector<bool> lane_trailing_cache_;

    std::uint64_t close_join_ns_ = 0;
    std::uint64_t close_drain_ns_ = 0;
    std::uint64_t close_host_ns_ = 0;
    std::uint64_t close_total_ns_ = 0;

    // ---- the per-plan RETAINED issue state (the SLOTS:
    //      depth-2 — two un-joined plans' state standing at once; the
    //      old engine's slot shape). Everything a lane enqueue still
    //      references — the operand-address arrays, the hook
    //      arguments, the host flags — lives in the slot until ITS
    //      close's barriers retire it.

    // The hook argument of one host-flag notify: the lane and the
    // handle (its address is handed to an in-flight notify, so the
    // slot's deque keeps it stable).
    struct HostFlagArg
    {
        HostWorkerLane *host;
        EventHandle handle;
    };

    // One instrumented op's stamp: the lane and the ONE leased stamp —
    // the clock read at the op's issue position — plus the queue
    // position the tile belongs to. `placed` is the sentinel (a handle
    // value is no sentinel: 0 is a lawful recycled handle); false
    // exactly for LINK/UNLINK and an untraced run. The handle lives
    // until THIS plan's close composes the op's tile and releases it —
    // the harvest's recycle.
    struct StampRef
    {
        LaneId lane = kNoLane;
        EventHandle read = 0;
        bool placed = false;
        std::size_t position = 0;
    };

    struct Slot
    {
        std::uint64_t plan_seq = 0;
        PlanDerivation deriv;      // this plan's owed sets (the join's
                                   // timing seam)
        std::vector<PlanOp> issue_log;
        std::vector<std::vector<OpRef>> issued_waits;
        std::vector<EventHandle> mint_handles; // per position, own mints
        std::vector<StampRef> stamp_refs;      // per position
        std::vector<EventHandle> tail_stamps;  // per lane: the issue
                                               // order's tail read (the
                                               // last tile's end)
        std::vector<char> tail_placed;         // per lane: the sentinel
                                               // (a handle value is no
                                               // sentinel — 0 is a
                                               // lawful recycled handle)
        std::unordered_map<OpRef, EventHandle> host_flags;
        std::deque<HostFlagArg> host_flag_args; // stable addresses
                                                 // while notifies fly
        // The operand pool: one flat per-slot arena replacing the
        // per-op vector allocation (the per-item cost cut — one heap
        // allocation per TILE_OP was the old path's floor). Bump-
        // allocated at issue, reset per plan; the slot lives until
        // the plan's close, so every array's bytes stay stable until
        // the barriers join them.
        std::vector<void *> operand_pool;
        std::size_t operand_pool_used = 0;
        std::vector<std::size_t> last_op_on_lane; // the barrier
                                                  // census's best name
        // THE TERMINAL MINTS (the previous engine's batch-granular
        // join, ported): one event handle per used lane, recorded at
        // the lane's tail at the ISSUE's end — the close joins THE
        // EVENT (cudaEventSynchronize), never the whole stream, so
        // the successor's already-fed prefix never delays the join
        // (the over-join that cost the pipelined loop its overlap).
        // The touched-lane test is last_op_on_lane, the same fact the
        // placement loop below uses — a handle value is no sentinel
        // (0 is a lawful recycled handle, the tail stamps' own rule).
        std::vector<EventHandle> terminal_handles;
    };
    std::vector<std::unique_ptr<Slot>> slots_; // dense; freed at close
    std::vector<std::size_t> free_slots_;

    // The slot an ISSUE (or a CLOSE) is currently driving: the helper
    // seams below (the host-flag notify, the wait matrix's flag
    // lookup, the dispatch's operand arrays, the census's issue
    // prefix) resolve their per-plan state through it. Engine-thread
    // only; null outside those two acts.
    Slot *cur_ = nullptr;

    // The queue close (after Slot, whose state it retires): reconcile,
    // barriers (device lanes first, host last — each device barrier
    // joins the bridge first), sweep + release, harvest, retire the
    // slot's buffers. The full PlanIssue witness lands in last_issue().
    void close_queue(Plan const &plan, PlanDerivation const &deriv,
        Slot &slot);

    // The most recently ISSUED plan's derivation (the pre-slot
    // surface, kept for the conformance suites).
    PlanDerivation last_derivation_;

    // (retention policy) The run-order history keeps the SCALAR
    // witness per plan (reconcile_ok above all — the meters fold it);
    // the newest plan's FULL payload (issue_log + issued_waits) lives
    // once, in last_full_ behind last_issue(), moved there instead of
    // deep-copied into the history at every close.
    std::vector<PlanIssue> issues_;
    PlanIssue last_full_;
    bool have_last_ = false;
    std::function<void(Plan const &, PlanDerivation const &)>
        post_derivation_hook_; // the documented test seam

    std::uint64_t ops_issued_ = 0;
    std::uint64_t mints_acquired_ = 0;
    std::uint64_t handles_released_ = 0;
    std::uint64_t last_seq_ = 0;
    bool have_seq_ = false;

    // The host-flag hook body (a driver thread or the host worker at a
    // lane position): one lock, one flag store, nothing else.
    static void set_host_flag_hook(void *arg) noexcept;
};

} // namespace nnhaul
