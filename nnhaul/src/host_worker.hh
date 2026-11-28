/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * host node's execution lane: one dedicated worker thread serving a
 * FIFO strictly in order — a lane SERVER, not a scheduler — plus the
 * trivial host event pool over the lane's flag array.
 *
 * @file src/host_worker.hh
 * HostWorkerLane — the host node's realization of ExecLane: TILE_OPs
 * execute in FIFO order, the worker calling the registered CPU function
 * on the operand addresses, and HostEventPool — mint/wait handles are
 * indices into the host lane's flag array (the same sizing rule as the
 * CUDA pools, conceptually; no CUDA events involved). The mint/wait
 * emulation runs BOTH directions with named costs: a DEVICE lane waits
 * a host mint by deferred enqueue or a gate host function blocking on
 * the flag (a blocked driver thread per gate — backend mechanism
 * reading flag(), never an engine query); the HOST lane waits a
 * device mint by gating the dependent work on the producer's notify
 * setting a host flag — a fire-ring-class callback
 * through set_flag(), never a poll. Stamps are the
 * worker's own clock reads at FIFO positions: the tracer's stamp
 * functions ride notify() items, and launch()/notify() are safely
 * usable with them from any thread — the worker IS the lane position.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include "backend.hh"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nnhaul
{

// The launch resolver — the registration-time stand-in until the host
// node's task table lands: the worker calls
// it in strict FIFO order to execute one TILE_OP's CPU body on the
// operand chunk addresses. It THROWS on a TaskId it resolves nothing
// for — a loud census, never a silent skip: the fault sticks,
// dependents never run, and it is delivered at the barrier.
using LaunchResolver = std::function<void(TaskId, void *const *)>;

// The trivial host EventPool — indices into the host
// lane's flag array, acquired and released, exhaustion loud. Lease
// discipline: acquire() and release() run on the engine thread only
// (the owed derivation acquires, the barrier sweep releases), so the
// free set carries no lock of its own.
class HostEventPool final : public EventPool
{
  public:
    // `flags` is the lane's flag array of `provision` entries; the
    // pool does not own it — the lane does.
    HostEventPool(
        std::atomic<std::uint32_t> *flags, std::size_t provision);

    ~HostEventPool() override = default;

    // Leases one index; exhaustion is the configuration fault,
    // loud with a census naming the provision. The belt: a leased
    // handle must carry a clear flag (release clears it), so an
    // unsound retirement is named, not silently reused.
    EventHandle acquire() override;

    // Returns one index to the provision and clears its flag — the
    // recycle. Retirement discipline (an entry dies only
    // when no current fact names the op) guarantees no waiter exists
    // at the sweep, so the clear races nobody. Out-of-provision and
    // double releases are bookkeeping defects, loud.
    void release(EventHandle e) override;

    // EventPool::join's host half: UNREACHED by construction — the
    // executor's batch-granular close joins device pools' terminal
    // events and the HOST lane keeps its FIFO barrier (its items are
    // host-side); a call here is an engine defect, loud.
    void join(EventHandle e) override;

    std::size_t provision() const { return provision_; }

    // The growth act's pool half, called only by the owning
    // lane's extend_provision under ITS mutex: rebind to the new flag
    // base and grow the lease bitmap. The bitmap RESIZES, never resets
    // — outstanding leases survive the growth. Lease discipline
    // unchanged: engine thread only.
    void rebase(
        std::atomic<std::uint32_t> *flags, std::size_t new_provision);

  private:
    std::atomic<std::uint32_t> *flags_;
    std::size_t provision_;
    std::vector<std::uint64_t> used_; // the lease bitmap, provision bits
    std::size_t used_count_ = 0;
    std::size_t hint_ = 0; // the word the first-fit scan restarts at
};

// The host node's execution lane. One worker thread
// serves a FIFO strictly in order (serialization): Launch items
// call the resolver on the operand addresses, Mint items store the
// event's flag (an in-order marker, host-visible alone), Gate items
// block the worker on the flag's condition variable
// until the flag is set (lane-internal mechanism: the worker IS
// the lane position, never an engine-plane poll), Notify
// items run their hook once at their position, and the barrier
// is the FIFO drain joined — raised only at queue close and run
// end, and it is the detection point: a faulted Launch sticks, its
// dependents never run, the mints behind it never complete, and the
// fault is thrown from barrier() with a census.
//
// Thread discipline: enqueue takes the lane mutex only to push (it
// never blocks on and never observes earlier work, and it does not
// read the fault state); the FIFO is unbounded and its growth is the
// one post-construction allocation, a harness fact; barrier signaling
// uses preallocated lane state, never a per-barrier allocation. The
// destructor is barrier + stop: it drains the FIFO (barrier's join,
// minus the throw), then stops and joins the worker; a gate still
// outstanding at teardown — the producer will never mint — is
// released into the fault path so the join cannot hang, and its
// dependents still never run.
class HostWorkerLane final : public ExecLane
{
  public:
    // One worker thread plus the event flag array of `event_provision`
    // handles, zeroed. A provision of zero is a
    // configuration fault, loud, before any resource exists. A null
    // `resolve` is allowed (registration may come later); a
    // Launch then faults loudly at its FIFO position.
    HostWorkerLane(std::size_t event_provision, LaunchResolver resolve);

    ~HostWorkerLane() override;

    //! Drain the FIFO into the teardown path but keep the lane alive
    //! (the bridge and the mint stream still need it while it drains).
    void drain_for_teardown();

    HostWorkerLane(HostWorkerLane const &) = delete;
    HostWorkerLane &operator=(HostWorkerLane const &) = delete;
    HostWorkerLane(HostWorkerLane &&) = delete;
    HostWorkerLane &operator=(HostWorkerLane &&) = delete;

    // A FIFO Launch. `addrs` is the engine-owned operand-address
    // array: the enqueue copies the POINTER, not the array (no
    // per-enqueue allocation), so the array must stay valid and
    // unchanged until the item is served — the host-lane equivalent of
    // the driver's launch-time argument capture, the buffer the
    // executor owns until the queue-close barrier retires it.
    void launch(TaskId task, NodeId node, void *const *addrs) override;

    // A FIFO Mint — the atomic flag store is the whole mint,
    // host-visible alone; minting never allocates.
    // `record`, when given, fires AT THIS FIFO POSITION instead of the
    // flag store: the engine binds a cudaEventRecord on
    // a dedicated host-mint stream, so a DEVICE consumer waits the
    // mint with a plain STREAM wait (cudaStreamWaitEvent), never a
    // deferral gate — sync between streams is the non-blocking stream wait
    // for event. A twin handle lives in a
    // DEVICE pool: its flag index is meaningless here (touched never)
    // and the twin's completion signal IS the event record. The hook
    // must not block (the worker IS the lane position); a throw faults
    // the lane and the bridge's recorded-wait parks release with
    // it (faulted(), below).
    using MintRecord = std::function<void(EventHandle)>;
    void mint(EventHandle e, MintRecord record);
    void mint(EventHandle e) override;

    // A FIFO Gate — later items do not run until the flag is set;
    // the worker blocks on the flag's condition variable, never a
    // spin, never a poll (the worker IS the lane position). An event
    // outside the provision is refused at submission (the first
    // detection point).
    void wait(EventHandle e) override;

    // Grow the flag provision (never shrink — loud): a
    // fresh array of the new size with the old flag VALUES copied in
    // (atomics do not move), swapped under mu_ — the worker
    // dereferences flags_ only under mu_, so parked gates re-index
    // into the new array on every wake — and the pool rebound. The
    // engine's growth commit calls this beside the device pools'
    // extend(); engine thread only.
    void extend_provision(std::size_t new_provision);

    // A FIFO Notify — the hook runs once, at its position. A null
    // hook is refused at submission, never a silent no-op.
    void notify(NotifyHook fn, void *arg) override;

    // ---- the stamp provision, the host lane's
    //      realization: the worker's OWN clock reads at FIFO
    //      positions. stamp_acquire/release run on
    //      the engine thread (the provision is the lane's own, grown
    //      to the working set and recycled at the harvests); stamp()
    //      is a FIFO Stamp item — submission records and returns,
    //      and the clock read happens AT the position, when
    //      the worker serves the item (exactly the semantics the
    //      notify-carried stamps had, minus a hook hop). stamp_read
    //      is the queue-close join's act, after the barrier drained
    //      the FIFO: the two slots' steady-clock reads, the ONE
    //      domain the trace renders.

    // Lease one stamp slot. Engine thread. Growth is the
    // harness-fact discipline (the FIFO's own); never serviced.
    EventHandle stamp_acquire() override;

    // Return one stamp slot — the harvest's act, after the pair was
    // read. Out-of-provision and double releases are bookkeeping
    // defects, loud, never serviced. Engine thread, under mu_ (a
    // successor plan's stamps may be in flight).
    void stamp_release(EventHandle h) override;

    // The FIFO Stamp item: the worker reads its own clock at the
    // position and writes the slot.
    void stamp(EventHandle h) override;

    // The pair's fact, post-barrier (engine thread, under mu_): the
    // before- and after-stamp slots. A slot never written is the
    // barrier's own defect — loud (the detection point is the join;
    // the FIFO drain guarantees every enqueued item ran).
    StampFact stamp_read(EventHandle before, EventHandle after) override;

    // The FIFO drain, joined — queue close and run end only. If a
    // fault stuck, THIS is the detection point: the census throws.
    void barrier() override;

    // The seam's host side, both directions. flag(e) hands the raw
    // atomic to the CUDA backend: a DEVICE lane waiting a
    // host mint blocks on it — the cudaLaunchHostFunc gate (a blocked
    // driver thread per gate; deferred enqueue is the other named
    // cost) — backend mechanism, never an engine query.
    std::atomic<std::uint32_t> *flag(EventHandle e);

    // The producer side of the seam: a lane's notify hook (the
    // fire-ring-class callback realizing a DEVICE mint on the host)
    // SETS a host flag through this — the store is taken under the
    // lane mutex with the gate condition variable notified, so a
    // gated worker wakes on it, never polls.
    void set_flag(EventHandle e);

    // The deferred-enqueue side of the seam (the other
    // named mechanism, realized by the backend-side deferral bridge
    // of cuda_backend.hh): parks the CALLER on this lane's own flag
    // machinery — the same cv_state_ the worker's own Gate items park
    // on, never a spin — until e's mint flag is host-visible, the
    // lane FAULTED, the lane tore down, or `cancel` turned true.
    // Returns true iff the flag is set; false names a release into
    // the fault/cancel path (a faulted host
    // lane never mints the flags behind the fault, so a parked gate
    // must be RELEASED to the fault path rather than wait forever —
    // the destructor's teardown discipline generalized to a live
    // lane). Backend mechanism, never an engine-plane query.
    bool wait_flag(EventHandle e, std::atomic<bool> const &cancel);

    // Wakes every parked wait_flag caller (predicates re-checked) —
    // the bridge's teardown act: its waiter parked on a flag whose
    // producer will never mint is released through its cancel.
    void kick();

    // The lane's fault state, read at JOINS (census material,
    // never an event query — the census is not an event lookup).
    bool faulted() const;

    // The FAULT NOTIFIER (the deadlock closure, made
    // deterministic): called once, at the fault, from the worker —
    // the bridge registers itself and releases every recorded-wait
    // park (a faulted worker skips the FIFO's twin-record items, so
    // their marks never land; the parks must wake). The hook runs
    // under mu_ — it must not block and must not re-enter the lane.
    void set_fault_notifier(std::function<void()> fn)
    {
        std::lock_guard<std::mutex> lock(mu_);
        fault_notifier_ = std::move(fn);
    }

    // Faults the lane from OUTSIDE — the bridge's fault propagation
    // (the destructor's fault-aware release
    // generalized to a live lane): a fault whose delivery parks on this
    // lane's flags — a DEVICE-lane fault killed the notify that would
    // have set them — releases every parked gate into the fault path
    // (the worker's own Gate items and any backend-side wait_flag
    // parks) and skips the FIFO's remaining items: dependents never
    // run, exactly as a lane-internal fault skips them. The census
    // carries `text`; never serviced.
    void fault(std::string text);

    // The trivial host EventPool over this lane's flag array.
    HostEventPool &pool() { return pool_; }

  private:
    // One FIFO entry: a kind tag plus the union of the five payloads —
    // trivially copyable, moved as bytes by the deque.
    struct Item
    {
        enum class Kind : std::uint8_t
        {
            Launch,
            Mint,
            Gate,
            Notify,
            Stamp,
            Barrier
        };
        Kind kind = Kind::Barrier;
        TaskId task = 0;                // Launch
        void *const *addrs = nullptr;   // Launch: engine-owned, stable
        EventHandle event = 0;          // Mint, Gate, Stamp
        NotifyHook fn = nullptr;        // Notify
        void *arg = nullptr;            // Notify
        MintRecord mint_record;         // Mint: the twin record hook
    };
    // (The Mint record hook made Item non-trivially-copyable — a
    // std::function rides the Mint items; the deque moves them fine.)

    void serve_();                       // the worker thread body
    void serve_launch_(Item const &it);  // runs the resolver, faults loud
    void push_(Item const &it);          // the enqueue
    void set_fault_locked_(std::string text); // mu_ held by the caller
    void require_event_(EventHandle e, char const *act) const;

    std::size_t provision_;             // the flag array's size
    LaunchResolver resolve_;
    std::vector<std::atomic<std::uint32_t>> flags_; // the provision
    HostEventPool pool_;

    // The stamp provision (the host realization):
    // the worker's clock reads at FIFO positions, steady ns, 0 = not
    // written (yet). Every access is under mu_ — the engine thread
    // leases, releases, and reads while a successor plan's stamps may
    // still be served; the close's barrier ordered this plan's slots
    // before its own pair read.
    std::vector<std::uint64_t> stamp_values_;
    std::vector<bool> stamp_leased_;
    std::vector<EventHandle> stamp_free_;

    std::deque<Item> fifo_;             // unbounded; growth is harness
                                        // fact (the one allocation)
    mutable std::mutex mu_; // faulted()'s join-time read locks it
    std::condition_variable cv_work_;   // FIFO push / pop / stop
    std::condition_variable cv_state_;  // flag transitions, the
                                        // barrier join's signal
    bool stop_ = false;      // set only after the FIFO drained
    bool teardown_ = false;  // the destructor: releases a stuck gate
    bool gate_done_ = true;  // the in-flight barrier's join signal
    bool faulted_ = false;   // worker-private; read at the join
    std::function<void()> fault_notifier_; // the bridge's release (mu_)
    std::string fault_;      // the fault's core census text
    std::uint64_t skipped_ = 0; // items skipped after the fault
    std::thread worker_;    // last member: joined in the destructor

  public:
    // The breakdown meters (the bridge accessors' siblings): the
    // worker's busy nanoseconds inside Launch items — the host
    // node's CPU bodies, the one place host TILE_OP time lands —
    // and the launch count. Relaxed; diagnosis only.
    std::uint64_t launch_busy_ns() const
    {
        return launch_busy_ns_.load(std::memory_order_relaxed);
    }
    std::uint64_t launch_count() const
    {
        return launch_count_.load(std::memory_order_relaxed);
    }

  private:
    std::atomic<std::uint64_t> launch_busy_ns_{0};
    std::atomic<std::uint64_t> launch_count_{0};
};

// ---- the backend-side deferral bridge (the deferred
//      enqueue, per lane) ----

// A gate parks ONLY ITS OWN LANE'S TAIL ("that
// lane's tail trails host progress"): one deferred queue and one
// waiter PER LANE that ever owed a host-family mint, every other
// lane's issue proceeding directly on the engine thread (issue stays
// non-blocking per lane). The global single-FIFO funnel — any gate
// parked the WHOLE issue, all lanes, one waiter thread — was a
// blocker; the gated mint family is the engine's own PUBLISH host-lane
// mint (the per-step acquire's publication, owed
// by the next spill-refetch COPY on the host->device channel).
//
// THE CAPTURE-SAFE DISCIPLINE (the per-lane cost, paid
// structurally): cudaStreamWaitEvent captures the event's state AT
// CALL TIME, so a DIRECT wait may run only past an EXECUTED record —
// a mint whose record still sits in a trailing lane's queue must
// defer the consumer's wait into the CONSUMER'S queue as a
// recorded-wait step, parking (a cv park, never a spin) until the
// record's OpRef enters the executed-record set; the consumer lane
// trails from that step. The mint-capture rule (every record call
// precedes its wait call) holds by construction:
// within a lane the queue IS the issue order, and the executed-record
// frontier only grows. No park can outlive host progress: a gate
// parks on the HOST LANE's own cv machinery (fault- and
// teardown-aware), a recorded-wait parks on a record that an EARLIER
// issue position queued on some lane — the issue-order induction
// (every park's dependency sits at an earlier position; each lane's
// queue and the host FIFO advance in order) keeps the frontier
// moving. THE HOST CONSUMER'S GATE NEVER RIDES THE BRIDGE at all:
// host_lane_->wait() pushes straight into the worker FIFO (a
// lane-internal park), and the flag-setting notify rides the
// producer's queue in issue order — the mixed-plan deadlock shape
// cannot re-form: the notify's enqueue precedes the gate's push
// on the engine thread, always.
//
// FAULTS, all classes, censused at the lane's close-join: a
// refused deferred step parks ITS lane's queue and surfaces at
// drain_lane (nothing after the faulting step executes — dependents
// never run); a HOST-lane fault releases every parked gate into the
// fault path (the destructor discipline generalized — wait_flag
// returns false, the lane's fault records the gate's tag); a
// DEVICE-lane async fault surfaces at the executor's terminal-event
// join of that lane (the light close's per-lane probe), the stuck
// parks releasing at teardown. NAMED RESIDUAL SEAM (one level down):
// a fault chain whose victim lane's drain
// precedes its killer's terminal join in the close order can park
// past the join — the exec lanes join first (the common killers);
// the cross-channel corner is teardown-released, recorded here.
// `tag` is the engine's step identity — the executor encodes the
// served op's OpRef, so the census names the exact op.
// wait_host_flag (above) stays landed as the not-chosen v1 gate —
// in practice a driver wedge under pipelined submission on this
// stack.
class DeferralBridge final
{
  public:
    // The waiter threads start lazily (one per lane that ever parks);
    // the host lane must outlive the bridge (the engine's member
    // order: bridge after the host lane, before the CUDA lanes).
    explicit DeferralBridge(HostWorkerLane &host);
    ~DeferralBridge(); // stop + kick + join every waiter

    DeferralBridge(DeferralBridge const &) = delete;
    DeferralBridge &operator=(DeferralBridge const &) = delete;
    DeferralBridge(DeferralBridge &&) = delete;
    DeferralBridge &operator=(DeferralBridge &&) = delete;

    // ---- the engine thread's acts — never blocking ----

    // One deferred enqueue for LANE's queue: executed in strict issue
    // order by that lane's waiter once the lane trails; direct on the
    // engine thread while it does not. `tag` is the served op's OpRef
    // (the fault census's name).
    void submit(LaneId lane, std::uint64_t tag, std::function<void()> fn);

    // A device-waits-host gate for LANE (the consumer's own lane):
    // parks the lane's queue head until the host mint flag is
    // host-visible (the host lane's cv machinery, fault-aware). From
    // this step the lane trails. `producer` names the lane the
    // flag-setting notify rides (the census's text).
    void gate(
        LaneId lane, std::uint64_t tag, EventHandle host_flag, LaneId producer);

    // The capture-safe wait: park the lane's queue until `mint`'s
    // record entered the executed set, then run `fn` (the stream
    // wait's enqueue). From this step the lane trails.
    void recorded_wait(LaneId lane, std::uint64_t tag, OpRef mint,
        std::function<void()> fn);

    // The executed-record frontier (see the class comment): true once
    // the mint's record was ENQUEUED — directly on the engine thread
    // or by some lane's waiter. mark_mint_recorded() never blocks.
    bool mint_recorded(OpRef mint) const;
    void mark_mint_recorded(OpRef mint);

    //! Drop recorded marks for plans older than `min_seq` (called after
    //! the close's registry sweep, when no event can wait on them).
    void forget_recorded_before(std::uint64_t min_seq);

    // True while LANE's queue holds steps (or its waiter is inside
    // one) — the engine's registration-order discipline keys on it:
    // a lane-table registration races that lane's waiter's launches
    // unless it rides the same queue.
    bool lane_trailing(LaneId lane) const;

    // Releases every recorded-wait park into the fault path (the
    // deadlock closure): the engine's record hook calls this
    // when the HOST lane faulted with twin records outstanding — the
    // parks lift with the census text, the drain delivers.
    void release_recorded_waits(std::string const &text);

    // ---- the close's per-lane join (the engine thread) ----

    // Block until LANE's queue is empty and its waiter idle — a park
    // bounded by host progress and record marks, never by device
    // completion (the gate flags are host-lane no-op mints; the
    // records are earlier issue positions) — then un-trail the lane
    // (its whole tail is enqueued; later direct enqueues land after
    // it in the stream, the stream's own order). Throws the lane's
    // fault if one stuck, naming the faulted step's tag. After the
    // drain the executor joins the lane's terminal event (the light
    // close) — never a stream sync.
    void drain_lane(LaneId lane);

    // The fault's facts, for the ENGINE's census wrap after a join
    // threw: the faulted step's tag (a gate carries its consumer's
    // ref, a refused call its own op's, 0 when none named) and text.
    bool faulted() const;
    std::uint64_t fault_tag() const;
    std::string const &fault_text() const;

    // The engine's close belt: true iff EVERY lane's queue is empty
    // and every waiter idle.
    bool idle() const;

    // The engine's close-side breakdown meters (read at shutdown):
    // the waiters' busy nanoseconds inside non-gate steps, the
    // nanoseconds PARKED at gates and record-waits, the step count,
    // and the close split (drain_lane's park vs the terminal joins
    // observed by the executor).
    std::uint64_t waiter_busy_ns() const
    {
        return waiter_busy_ns_.load(std::memory_order_relaxed);
    }
    std::uint64_t waiter_parked_ns() const
    {
        return waiter_parked_ns_.load(std::memory_order_relaxed);
    }
    std::uint64_t waiter_steps() const
    {
        return waiter_steps_.load(std::memory_order_relaxed);
    }
    std::uint64_t gates_parked() const
    {
        return gates_parked_.load(std::memory_order_relaxed);
    }
    std::uint64_t record_waits() const
    {
        return record_waits_.load(std::memory_order_relaxed);
    }

  private:
    // One deferred step: a host gate, a plain call, or a
    // capture-safe record-wait — tagged with the served op's ref.
    struct Step
    {
        enum class Kind : std::uint8_t
        {
            Gate,
            Call,
            RecordWait
        };
        Kind kind = Kind::Call;
        std::uint64_t tag = 0;     // the served op's OpRef
        EventHandle gate = 0;      // Gate: the host mint flag
        LaneId producer = kNoLane; // Gate: the notify's lane (census)
        OpRef mint = 0;            // RecordWait: the awaited record
        std::function<void()> call; // Call, RecordWait's body
    };

    // One lane's deferred state: the queue, its waiter, and its
    // fault. Created lazily on the lane's first parked step; the map
    // key is the LaneId. The waiter thread's lifetime is the lane
    // state's — joined at teardown.
    struct LaneState
    {
        std::mutex mu;
        std::condition_variable cv_work; // work arrived / stop
        std::condition_variable cv_done; // step retired — the drain
        std::deque<Step> steps;
        bool trailing = false;   // steps exist / waiter busy
        bool executing = false;  // the waiter is inside a step
        bool faulted = false;
        std::uint64_t fault_tag = 0;
        std::string fault_text;
        std::thread waiter;
    };

    LaneState *lane_state_(LaneId lane); // lazily created, never null
    void serve_(LaneId lane, LaneState *st);

    // The fault-aware release, per parked lane: record the fault and
    // release the parks — for a HOST fault the lane's cv park already
    // lifted (wait_flag's fault predicate); this records the census.
    void release_lane_(LaneState &st, std::uint64_t tag,
        std::string const &text);

    HostWorkerLane *host_;

    // The executed-record frontier (capture safety): the set of mint
    // OpRefs whose record was enqueued, and the cv its waiters park
    // on. One bridge-wide mutex guards the set, the lane map, and
    // every LaneState's cross-thread fields (each LaneState's own mu
    // guards its queue; taking both is always bridge-mu first).
    mutable std::mutex mu_;
    std::condition_variable cv_recorded_; // a record entered the set
    std::unordered_set<OpRef> recorded_;
    std::unordered_map<LaneId, std::unique_ptr<LaneState>> lanes_;
    // The engine-thread cache over `lanes_` (dense lane ids): the
    // hot path (submit/gate/recorded_wait) resolves a lane's state
    // with one indexed read instead of mu_ + a map find. Written and
    // read by the ENGINE THREAD only — every entry's writer is a
    // lane_state_ call from that thread; cross-thread readers keep
    // using `lanes_` under mu_.
    std::vector<LaneState *> lane_by_id_;
    bool stop_ = false;            // guarded by mu_
    bool any_faulted_ = false;     // guarded by mu_
    bool host_faulted_ = false;    // guarded by mu_; the lane's
                                   // notifier publishes it (the cv
                                   // predicate reads it WITHOUT the
                                   // lane's mu_ — no lock-order cycle)
    std::uint64_t fault_tag_ = 0;  // the first fault's census facts
    std::string fault_text_;

    // The meters (relaxed — diagnosis counters, never control flow).
    std::atomic<std::uint64_t> waiter_busy_ns_{0};
    std::atomic<std::uint64_t> waiter_parked_ns_{0};
    std::atomic<std::uint64_t> waiter_steps_{0};
    std::atomic<std::uint64_t> gates_parked_{0};
    std::atomic<std::uint64_t> record_waits_{0};
    std::atomic<bool> stop_flag_{false}; // the destructor's kick flag
};


} // namespace nnhaul} // namespace nnhaul
