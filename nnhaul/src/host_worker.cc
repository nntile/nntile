/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * host worker lane's acts: the worker thread's FIFO service, the five
 * enqueues, the both-direction mint/wait seam, and the trivial
 * host event pool's lease discipline.
 *
 * @file src/host_worker.cc
 * The acts behind src/host_worker.hh: the
 * worker loop serving Launch, Mint, Gate, Notify, and Barrier items
 * strictly in FIFO order; mint as the atomic flag store
 * host-visible alone; the gate's condition-variable
 * block on the flag array (lane-internal mechanism, the worker IS the
 * lane position, never a poll); the once-executed notify
 * at its position; the barrier drain joined and the fault census
 * delivered at it; and HostEventPool's acquire/release with
 * the exhaustion census .
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#include "host_worker.hh"

#include <chrono>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{
// The breakdown meters' clock (diagnosis only, never control flow):
// steady nanoseconds, local to this vendor-free file.
std::uint64_t now_ns_local()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}
} // namespace


namespace nnhaul
{

namespace
{

// A provision of zero is a configuration fault; validating before any
// member constructs keeps the failure clean (no thread to join).
std::size_t checked_provision(std::size_t n)
{
    if (n == 0)
    {
        throw std::runtime_error(
            "host lane: an event provision of zero is a configuration "
            "fault (see the host event sizing; NNHAUL_EVENT_POOL "
            "overriding)");
    }
    return n;
}

} // namespace

// ---- HostEventPool  ----

HostEventPool::HostEventPool(
    std::atomic<std::uint32_t> *flags, std::size_t provision)
    : flags_(flags),
      provision_(provision),
      used_((provision + 63u) / 64u, 0u)
{
}

EventHandle HostEventPool::acquire()
{
    for (std::size_t w = 0; w < used_.size(); ++w)
    {
        std::size_t const wi = (hint_ + w) % used_.size();
        // The word's VALID bits: the provision's tail word carries
        // bits past the provision that no handle names.
        std::uint64_t const valid =
            (wi + 1u == used_.size() && provision_ % 64u != 0u)
            ? ((static_cast<std::uint64_t>(1) << (provision_ % 64u)) -
                  1u)
            : ~static_cast<std::uint64_t>(0);
        std::uint64_t const free_bits = ~used_[wi] & valid;
        if (free_bits == 0)
        {
            continue;
        }
        std::size_t b = 0;
        while ((free_bits & (static_cast<std::uint64_t>(1) << b)) == 0)
        {
            ++b;
        }
        used_[wi] |= static_cast<std::uint64_t>(1) << b;
        ++used_count_;
        hint_ = wi;
        EventHandle const e = static_cast<EventHandle>(wi * 64u + b);
        // The belt: a handle enters its lease with a clear flag, and
        // release clears — a live mint on a fresh lease names an
        // unsound retirement, never a silently reused handle.
        if (flags_[e].load(std::memory_order_acquire) != 0u)
        {
            throw std::runtime_error(
                "host event pool: event " + std::to_string(e) +
                " leased with its mint flag live — the retirement "
                "rule was broken");
        }
        return e;
    }
    // Exhaustion is a configuration fault, loud with a census.
    throw std::runtime_error(
        "host event pool exhausted: configuration fault — provision " +
        std::to_string(provision_) + " handles, " +
        std::to_string(used_count_) + " leased; the host provision "
        "follows the host event sizing (pow2ceil(2 * max_p M_p) "
        "floored at 1024, NNHAUL_EVENT_POOL overriding) — raise it at "
        "init");
}

void HostEventPool::rebase(
    std::atomic<std::uint32_t> *flags, std::size_t new_provision)
{
    flags_ = flags;
    provision_ = new_provision;
    used_.resize((new_provision + 63u) / 64u, 0u);
}

void HostEventPool::release(EventHandle e)
{
    std::size_t const at = static_cast<std::size_t>(e);
    if (at >= provision_)
    {
        throw std::runtime_error(
            "host event pool: release of event " + std::to_string(e) +
            " outside the provision " + std::to_string(provision_) +
            " — a bookkeeping defect");
    }
    std::uint64_t const bit =
        static_cast<std::uint64_t>(1) << (at % 64u);
    if ((used_[at / 64u] & bit) == 0)
    {
        throw std::runtime_error(
            "host event pool: release of the unleased event " +
            std::to_string(e) + " — a bookkeeping defect (a double "
            "release)");
    }
    used_[at / 64u] &= ~bit;
    --used_count_;
    // The recycle: the flag is cleared so the handle's next lease
    // mints from zero; no waiter exists at the sweep .
    flags_[at].store(0u, std::memory_order_release);
}

// ---- HostWorkerLane: construction and the join of its life ----

HostWorkerLane::HostWorkerLane(
    std::size_t event_provision, LaunchResolver resolve)
    : provision_(checked_provision(event_provision)),
      resolve_(std::move(resolve)),
      flags_(provision_),
      pool_(flags_.data(), provision_),
      worker_([this] { serve_(); })
{
    // The provision starts clear, explicitly.
    for (std::size_t i = 0; i < provision_; ++i)
    {
        flags_[i].store(0u, std::memory_order_relaxed);
    }
}

void HostWorkerLane::drain_for_teardown()
{
    {
        std::lock_guard<std::mutex> lock(mu_);
        // Barrier + stop: the drain below joins the FIFO's tail; the
        // teardown flag releases a gate whose producer will never
        // mint — into the fault path, so the join cannot hang and the
        // dependents still never run.
        teardown_ = true;
        cv_work_.notify_all();
        cv_state_.notify_all();
    }
    std::unique_lock<std::mutex> lock(mu_);
    cv_work_.wait(lock, [this] { return fifo_.empty(); });
}

HostWorkerLane::~HostWorkerLane()
{
    drain_for_teardown();
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
        cv_work_.notify_all();
    }
    worker_.join();
}

// ---- the enqueues (push and return) ----

void HostWorkerLane::push_(Item const &it)
{
    // The mutex is held only for the push itself — never against
    // earlier work, and the enqueues read no lane state.
    std::lock_guard<std::mutex> lock(mu_);
    fifo_.push_back(it);
    cv_work_.notify_all();
}

void HostWorkerLane::require_event_(
    EventHandle e, char const *act) const
{
    if (static_cast<std::size_t>(e) >= provision_)
    {
        // A submission error the lane refuses (the first detection
        // point), never a silent skip.
        throw std::runtime_error(std::string("host lane: ") + act +
            " on event " + std::to_string(e) + " outside the provision " +
            std::to_string(provision_) + " — refused at submission");
    }
}

void HostWorkerLane::launch(
    TaskId task, NodeId node, void *const *addrs)
{
    (void)node; // the host node's own lane — the engine's lane_of_exec
                // routed the TILE_OP here; the node id
                // carries no further fact for the host lane.
    Item it;
    it.kind = Item::Kind::Launch;
    it.task = task;
    it.addrs = addrs;
    push_(it);
}

void HostWorkerLane::mint(EventHandle e)
{
    mint(e, MintRecord{});
}

void HostWorkerLane::mint(EventHandle e, MintRecord record)
{
    if (record)
    {
        // A twin mint: the handle lives in a DEVICE pool — the event
        // index is not a flag index, so the flag-store path never
        // touches it (the collision would break the host pool's lease
        // belt). The record hook fires at this FIFO position.
        Item it;
        it.kind = Item::Kind::Mint;
        it.event = e;
        it.mint_record = std::move(record);
        push_(it);
        return;
    }
    require_event_(e, "mint");
    Item it;
    it.kind = Item::Kind::Mint;
    it.event = e;
    push_(it);
}

void HostWorkerLane::wait(EventHandle e)
{
    require_event_(e, "wait");
    Item it;
    it.kind = Item::Kind::Gate;
    it.event = e;
    push_(it);
}

void HostWorkerLane::extend_provision(std::size_t new_provision)
{
    if (new_provision <= provision_)
    {
        return; // the provision is a monotone floor
    }
    // Under mu_: the worker dereferences flags_ only under mu_, so a
    // parked gate re-indexes into the new array on every wake. The
    // VALUES copy is element-wise (atomics do not move); the swap is a
    // buffer exchange, and the pool rebinds to the new base with its
    // bitmap grown, leases intact.
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::atomic<std::uint32_t>> grown(new_provision);
    for (std::size_t i = 0; i < provision_; ++i)
    {
        grown[i].store(
            flags_[i].load(std::memory_order_relaxed),
            std::memory_order_relaxed);
    }
    flags_.swap(grown);
    provision_ = new_provision;
    pool_.rebase(flags_.data(), new_provision);
}

void HostWorkerLane::notify(NotifyHook fn, void *arg)
{
    if (fn == nullptr)
    {
        throw std::runtime_error(
            "host lane: notify of a null hook refused at submission");
    }
    Item it;
    it.kind = Item::Kind::Notify;
    it.fn = fn;
    it.arg = arg;
    push_(it);
}

// ---- the stamp provision, the host lane's realization ----

EventHandle HostWorkerLane::stamp_acquire()
{
    std::lock_guard<std::mutex> lock(mu_);
    if (stamp_free_.empty())
    {
        // The growth batch: the provision grows to the working set
        // (the un-retired plans' stamps) and recycles at the
        // harvests — the FIFO's own harness-fact discipline. Zero is
        // "not written (yet)": steady ns since epoch is far past it.
        constexpr std::size_t kBatch = 256;
        std::size_t const base = stamp_values_.size();
        stamp_values_.resize(base + kBatch, 0);
        stamp_leased_.resize(base + kBatch, true);
        for (std::size_t i = 0; i < kBatch; ++i)
        {
            stamp_free_.push_back(static_cast<EventHandle>(base + i));
        }
    }
    EventHandle const h = stamp_free_.back();
    stamp_free_.pop_back();
    stamp_leased_[static_cast<std::size_t>(h)] = true; // the recycle
                                                       // re-leases
    return h;
}

void HostWorkerLane::stamp_release(EventHandle h)
{
    std::lock_guard<std::mutex> lock(mu_);
    std::size_t const at = static_cast<std::size_t>(h);
    if (at >= stamp_values_.size())
    {
        throw std::runtime_error(
            "host lane: stamp_release of stamp " + std::to_string(h) +
            " outside the provision " +
            std::to_string(stamp_values_.size()) +
            " — a bookkeeping defect");
    }
    if (!stamp_leased_[at])
    {
        throw std::runtime_error(
            "host lane: stamp_release of the unleased stamp " +
            std::to_string(h) + " — a bookkeeping defect (a double "
            "release)");
    }
    stamp_leased_[at] = false;
    stamp_values_[at] = 0;
    stamp_free_.push_back(h);
}

void HostWorkerLane::stamp(EventHandle h)
{
    // Submission: the FIFO Stamp item — the clock read happens AT
    // the position, when the worker serves it .
    std::size_t const at = static_cast<std::size_t>(h);
    std::lock_guard<std::mutex> lock(mu_);
    if (at >= stamp_values_.size() || !stamp_leased_[at])
    {
        throw std::runtime_error(
            "host lane: stamp record of stamp " + std::to_string(h) +
            " outside the leased provision — refused at submission");
    }
    Item it;
    it.kind = Item::Kind::Stamp;
    it.event = h;
    fifo_.push_back(it);
    cv_work_.notify_all();
}

StampFact HostWorkerLane::stamp_read(EventHandle before, EventHandle after)
{
    // The queue-close join's read : the barrier
    // drained the FIFO, so both slots carry their position's clock
    // reads. A slot of zero is the barrier's own defect — loud (the
    // detection point is the join), never serviced.
    std::lock_guard<std::mutex> lock(mu_);
    std::size_t const b = static_cast<std::size_t>(before);
    std::size_t const a = static_cast<std::size_t>(after);
    if (b >= stamp_values_.size() || a >= stamp_values_.size())
    {
        throw std::runtime_error(
            "host lane: stamp read of a handle outside the provision "
            "— a bookkeeping defect");
    }
    if (stamp_values_[b] == 0 || stamp_values_[a] == 0)
    {
        throw std::runtime_error(
            "host lane: a stamp slot of the pair is unwritten at the "
            "queue-close join — the FIFO drain did not retire the "
            "position , a barrier defect");
    }
    StampFact fact;
    fact.start_ns = stamp_values_[b];
    fact.end_ns = stamp_values_[a];
    return fact;
}

void HostWorkerLane::barrier()
{
    std::unique_lock<std::mutex> lock(mu_);
    gate_done_ = false;
    Item barrier_item;
    barrier_item.kind = Item::Kind::Barrier;
    fifo_.push_back(barrier_item);
    cv_work_.notify_all();
    cv_state_.wait(lock, [this] { return gate_done_; });
    if (faulted_)
    {
        // The queue-close barrier IS the detection point — the
        // fault is delivered here, with its census; a fault is never
        // serviced and the run stops.
        std::string const text = fault_ + "; " +
            std::to_string(skipped_) + " later FIFO items skipped "
            "(dependents never run) and any mint behind the fault "
            "never completed";
        lock.unlock();
        throw std::runtime_error(text);
    }
}

// ---- the seam  ----

std::atomic<std::uint32_t> *HostWorkerLane::flag(EventHandle e)
{
    require_event_(e, "flag");
    return &flags_[e];
}

void HostWorkerLane::set_flag(EventHandle e)
{
    require_event_(e, "set_flag");
    std::lock_guard<std::mutex> lock(mu_);
    flags_[e].store(1u, std::memory_order_release);
    cv_state_.notify_all();
}

bool HostWorkerLane::wait_flag(
    EventHandle e, std::atomic<bool> const &cancel)
{
    // The deferred-enqueue park : the worker's own Gate
    // item's mechanism generalized to a backend-side caller — one
    // condition-variable park on the flag, the fault state, and the
    // caller's cancel, never a spin (the caller is
    // backend mechanism, never an engine query).
    require_event_(e, "wait_flag");
    std::unique_lock<std::mutex> lock(mu_);
    cv_state_.wait(lock, [this, e, &cancel] {
        return teardown_ || faulted_ ||
            flags_[e].load(std::memory_order_acquire) != 0u ||
            cancel.load(std::memory_order_acquire);
    });
    return !teardown_ && !faulted_ &&
        !cancel.load(std::memory_order_acquire) &&
        flags_[e].load(std::memory_order_acquire) != 0u;
}

bool HostWorkerLane::faulted() const
{
    std::lock_guard<std::mutex> lock(mu_);
    return faulted_;
}

void HostWorkerLane::kick()
{
    // Every parked wait_flag re-checks its predicate; a cancel turned
    // true releases its park (the bridge's teardown act).
    std::lock_guard<std::mutex> lock(mu_);
    cv_state_.notify_all();
}

void HostWorkerLane::fault(std::string text)
{
    // The external fault seam (the bridge's fault propagation): the
    // destructor's fault-aware release generalized to a live lane —
    // every
    // parked gate releases into the fault path (the worker's own Gate
    // items through cv_state_, backend-side wait_flag parks likewise)
    // and the FIFO's remaining items are skipped: dependents never
    // run. The census carries the text; never serviced.
    std::lock_guard<std::mutex> lock(mu_);
    set_fault_locked_(std::move(text));
}

// ---- the worker (the lane SERVER, not a scheduler) ----

void HostWorkerLane::set_fault_locked_(std::string text)
{
    if (!faulted_)
    {
        faulted_ = true; // the first fault sticks
        fault_ = std::move(text);
        // The fault-aware release (the destructor discipline
        // generalized): every gate parked on this lane's mint flags —
        // the worker's own Gate items behind the fault AND any
        // backend-side wait_flag parks — is released into the fault
        // path, never left waiting on a mint that will never fire.
        cv_state_.notify_all();
        if (fault_notifier_)
        {
            // The bridge's recorded-wait parks wake the same way: a
            // faulted worker SKIPS the FIFO's twin-record items, so
            // their marks never land (the deadlock is closed
            // deterministically — the predicate reads faulted(), the
            // notifier supplies the wake).
            fault_notifier_();
        }
    }
}

void HostEventPool::join(EventHandle e)
{
    (void) e;
    throw std::runtime_error(
        "host event pool: the batch-granular join of a host-family "
        "event is unreachable — the host lane's FIFO barrier is the "
        "host side's join (an engine defect, loud)");
}

void HostWorkerLane::serve_launch_(Item const &it)
{
    std::uint64_t const t0 = now_ns_local();
    try
    {
        if (!resolve_)
        {
            throw std::runtime_error(
                "no launch resolver registered — the host node's task "
                "table is absent ");
        }
        resolve_(it.task, it.addrs);
    }
    catch (std::exception const &ex)
    {
        std::lock_guard<std::mutex> lock(mu_);
        set_fault_locked_("host-lane fault at a Launch of task " +
            std::to_string(it.task) + ": " + ex.what());
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lock(mu_);
        set_fault_locked_("host-lane fault at a Launch of task " +
            std::to_string(it.task) + ": an unknown fault");
    }
    launch_busy_ns_.fetch_add(
        now_ns_local() - t0, std::memory_order_relaxed);
    launch_count_.fetch_add(1, std::memory_order_relaxed);
}

void HostWorkerLane::serve_()
{
    for (;;)
    {
        Item it;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_work_.wait(lock, [this] { return stop_ || !fifo_.empty(); });
            if (fifo_.empty())
            {
                return; // stop_ with the FIFO drained
            }
            it = fifo_.front();
        }
        // Once faulted, dependents never run — the item is popped
        // and skipped, so a mint behind the fault never completes; a
        // Barrier item still signals its join, the fault's delivery.
        if (faulted_ && it.kind != Item::Kind::Barrier)
        {
            std::lock_guard<std::mutex> lock(mu_);
            ++skipped_;
            fifo_.pop_front();
            if (fifo_.empty())
            {
                cv_work_.notify_all();
            }
            continue;
        }
        switch (it.kind)
        {
        case Item::Kind::Launch:
            serve_launch_(it);
            break;
        case Item::Kind::Mint:
        {
            if (it.mint_record)
            {
                // The twin record, AT THIS FIFO POSITION: a plain event
                // record on the engine-bound stream (never blocking —
                // the worker IS the lane position). A throw faults the
                // lane; the bridge's recorded-wait parks release
                // through faulted().
                try
                {
                    MintRecord record = std::move(it.mint_record);
                    record(it.event);
                }
                catch (std::exception const &ex)
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    set_fault_locked_(
                        std::string("host-lane fault at a Mint's "
                                    "record twin: ") +
                        ex.what());
                }
                break;
            }
            std::lock_guard<std::mutex> lock(mu_);
            // An in-order marker — the atomic store IS
            // the mint, host-visible alone; the notify covers every
            // waiter class uniformly (a gate on this lane's own mint
            // is elided by the engine, never issued).
            flags_[it.event].store(1u, std::memory_order_release);
            cv_state_.notify_all();
            break;
        }
        case Item::Kind::Gate:
        {
            std::unique_lock<std::mutex> lock(mu_);
            // The worker blocks on the flag's condition variable —
            // lane-internal mechanism (the worker IS the lane
            // position), never an engine-plane poll. Later FIFO items
            // do not run until the flag is set;
            // the mutex is released while blocked, so submission stays
            // non-blocking.
            cv_state_.wait(lock, [this, &it] {
                return teardown_ ||
                    flags_[it.event].load(std::memory_order_acquire) !=
                        0u;
            });
            if (flags_[it.event].load(std::memory_order_acquire) == 0u)
            {
                set_fault_locked_(
                    "host-lane teardown with a gate on event " +
                    std::to_string(it.event) +
                    " outstanding: the producer never minted");
            }
            break;
        }
        case Item::Kind::Notify:
            // Once, at this FIFO position. The host-flag gates
            // ride this hook.
            it.fn(it.arg);
            break;
        case Item::Kind::Stamp:
        {
            // The stamp provision, the host realization: the worker's
            // own clock read AT this FIFO position,
            // written into the leased slot. The close's barrier
            // orders it before the pair read; mu_ guards against a
            // successor plan's concurrent lease/release.
            std::uint64_t const ns =
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now()
                            .time_since_epoch())
                        .count());
            std::lock_guard<std::mutex> lock(mu_);
            stamp_values_[static_cast<std::size_t>(it.event)] = ns;
            break;
        }
        case Item::Kind::Barrier:
        {
            std::lock_guard<std::mutex> lock(mu_);
            gate_done_ = true;
            cv_state_.notify_all();
            break;
        }
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            fifo_.pop_front();
            if (fifo_.empty())
            {
                cv_work_.notify_all();
            }
        }
    }
}

// ---- DeferralBridge  ----

namespace
{
std::uint64_t bridge_now_ns()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}
}

DeferralBridge::DeferralBridge(HostWorkerLane &host) : host_(&host)
{
    // The fault-notifier wiring (the deadlock closure): the
    // host lane's fault releases this bridge's recorded-wait parks —
    // a faulted worker skips the twin-record items, so the parks'
    // awaited marks never land. The hook runs under the lane's mu_:
    // release_recorded_waits takes this bridge's mu_ and never
    // re-enters the lane (lock order: lane mu_ -> bridge mu_ — the
    // reverse never happens: the bridge's waiters take bridge mu_
    // first and only TOUCH host_->faulted() (a read under lane mu_)
    // while NOT holding bridge mu_... they hold it inside the cv
    // lambda — lane mu_ is taken there. DEADLOCK RISK: a waiter holds
    // bridge mu_ inside cv_recorded_.wait's predicate and takes lane
    // mu_ (faulted()); this hook holds lane mu_ and takes bridge mu_.
    // Fix: the predicate reads faulted() via a cached atomic the hook
    // publishes — see host_faulted_ below.
    host.set_fault_notifier([this]() {
        std::lock_guard<std::mutex> lock(mu_);
        host_faulted_ = true;
        cv_recorded_.notify_all();
    });
}

DeferralBridge::~DeferralBridge()
{
    // Stop + kick + join every waiter: a waiter parked on a host flag
    // releases through its cancel (wait_flag's predicate re-checks
    // after the kick), a parked FAULT already recorded its census —
    // either way no join can hang. The host lane outlives the bridge
    // (the engine's member order: bridge after the host lane).
    stop_flag_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
        cv_recorded_.notify_all();
    }
    host_->kick();
    for (auto &kv : lanes_)
    {
        std::lock_guard<std::mutex> lock(kv.second->mu);
        kv.second->cv_work.notify_all();
        kv.second->cv_done.notify_all();
    }
    for (auto &kv : lanes_)
    {
        if (kv.second->waiter.joinable())
        {
            kv.second->waiter.join();
        }
    }
}

DeferralBridge::LaneState *DeferralBridge::lane_state_(LaneId lane)
{
    // Lazily created at the lane's first parked step; the waiter
    // thread starts with the state. mu_ guards the map. The hot
    // path: lane_state_ and every trailing-flag writer (gate,
    // recorded_wait, drain_lane) run on the ENGINE THREAD ONLY, so
    // the by-id cache is written and read by that one thread — a
    // plain vector beats the map lookup + mu_ per submit, and no
    // lock is sound for it (the map itself stays the owner, guarded
    // as ever for the cross-thread readers: serve_ holds the state
    // pointer from creation, idle()/teardown take mu_).
    if (lane < lane_by_id_.size() && lane_by_id_[lane] != nullptr)
    {
        return lane_by_id_[lane];
    }
    std::lock_guard<std::mutex> lock(mu_);
    auto it = lanes_.find(lane);
    if (it != lanes_.end())
    {
        if (lane >= lane_by_id_.size())
        {
            lane_by_id_.resize(lane + 1u, nullptr);
        }
        lane_by_id_[lane] = it->second.get();
        return it->second.get();
    }
    std::unique_ptr<LaneState> st = std::make_unique<LaneState>();
    LaneState *raw = st.get();
    raw->waiter = std::thread([this, lane, raw] { serve_(lane, raw); });
    lanes_.emplace(lane, std::move(st));
    if (lane >= lane_by_id_.size())
    {
        lane_by_id_.resize(lane + 1u, nullptr);
    }
    lane_by_id_[lane] = raw;
    return raw;
}

void DeferralBridge::submit(
    LaneId lane, std::uint64_t tag, std::function<void()> fn)
{
    // The engine thread's enqueue — direct while THIS lane trails
    // nothing, queued in the lane's own issue order once it
    // does. Only the lane's tail ever defers: every other lane keeps
    // issuing on the engine thread. The fast path: `trailing` is
    // written only by the engine thread (gate, recorded_wait,
    // drain_lane — the same thread that reads it here), so the
    // direct call needs NO lock at all — directness is lawful
    // because the flag is single-writer and this
    // reader is that same thread.
    LaneState *st = lane_state_(lane);
    if (!st->trailing)
    {
        fn();
        return;
    }
    std::unique_lock<std::mutex> lock(st->mu);
    st->steps.push_back(Step{Step::Kind::Call, tag, 0, kNoLane, 0,
        std::move(fn)});
    st->cv_work.notify_all();
}

void DeferralBridge::gate(LaneId lane, std::uint64_t tag,
    EventHandle host_flag, LaneId producer)
{
    // A device-waits-host gate parks ONLY this lane's queue head: the
    // flag lifts with host progress (the host lane's cv machinery —
    // fault- and teardown-aware, never a spin), and the lane trails
    // from this step. `producer` names the notify's lane for the
    // census text.
    LaneState *st = lane_state_(lane);
    std::lock_guard<std::mutex> lock(st->mu);
    st->trailing = true;
    st->steps.push_back(Step{Step::Kind::Gate, tag, host_flag, producer,
        0, {}});
    st->cv_work.notify_all();
}

void DeferralBridge::recorded_wait(LaneId lane, std::uint64_t tag,
    OpRef mint, std::function<void()> fn)
{
    // The capture-safe wait: the mint's record is not enqueued yet
    // (it sits at an EARLIER issue position in some trailing lane's
    // queue), so the consumer's stream wait must not run before it.
    // The step parks on the executed-record frontier's cv — bounded
    // by that earlier position's own progress, never device
    // completion — then enqueues the wait. The lane trails from here.
    LaneState *st = lane_state_(lane);
    std::lock_guard<std::mutex> lock(st->mu);
    st->trailing = true;
    st->steps.push_back(Step{Step::Kind::RecordWait, tag, 0, kNoLane,
        mint, std::move(fn)});
    st->cv_work.notify_all();
}

bool DeferralBridge::mint_recorded(OpRef mint) const
{
    std::lock_guard<std::mutex> lock(mu_);
    return recorded_.count(mint) != 0;
}

void DeferralBridge::mark_mint_recorded(OpRef mint)
{
    {
        std::lock_guard<std::mutex> lock(mu_);
        recorded_.insert(mint);
    }
    cv_recorded_.notify_all();
}

void DeferralBridge::forget_recorded_before(std::uint64_t min_seq)
{
    std::lock_guard<std::mutex> lock(mu_);
    for (auto it = recorded_.begin(); it != recorded_.end();)
    {
        if ((*it >> 32) < min_seq)
        {
            it = recorded_.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

bool DeferralBridge::lane_trailing(LaneId lane) const
{
    std::lock_guard<std::mutex> lock(mu_);
    auto it = lanes_.find(lane);
    if (it == lanes_.end())
    {
        return false;
    }
    std::lock_guard<std::mutex> lane_lock(it->second->mu);
    return it->second->trailing;
}

void DeferralBridge::release_recorded_waits(std::string const &text)
{
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!any_faulted_)
        {
            any_faulted_ = true;
            fault_tag_ = 0;
            fault_text_ = text;
        }
        cv_recorded_.notify_all();
    }
}

void DeferralBridge::drain_lane(LaneId lane)
{
    // The close's per-lane join: block until the lane's queue is
    // empty and its waiter idle — a park bounded by host progress and
    // record marks, never device completion — then un-trail the lane
    // (its whole tail is enqueued; the stream's own order carries
    // later direct enqueues behind it). A stuck fault throws with the
    // faulted step's tag; the bridge-level facts (faulted()/
    // fault_tag()/fault_text()) are set here for the ENGINE's census
    // wrap, and the record-wait parks release with them.
    LaneState *st = lane_state_(lane);
    std::uint64_t tag = 0;
    std::string text;
    bool faulted = false;
    {
        std::unique_lock<std::mutex> lock(st->mu);
        st->cv_done.wait(lock, [&st] {
            return st->faulted || (st->steps.empty() && !st->executing);
        });
        if (st->faulted)
        {
            faulted = true;
            tag = st->fault_tag;
            text = st->fault_text;
        }
        else
        {
            st->trailing = false;
        }
    }
    if (faulted)
    {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!any_faulted_)
            {
                any_faulted_ = true;
                fault_tag_ = tag;
                fault_text_ = text;
            }
            cv_recorded_.notify_all(); // release the record-wait parks
        }
        throw std::runtime_error(
            "deferral bridge (lane " + std::to_string(lane) +
            ") faulted: " + text + " (the faulted step's tag " +
            std::to_string(tag) +
            "; nothing after it executed — dependents never run)");
    }
}

bool DeferralBridge::faulted() const
{
    std::lock_guard<std::mutex> lock(mu_);
    return any_faulted_;
}

std::uint64_t DeferralBridge::fault_tag() const
{
    std::lock_guard<std::mutex> lock(mu_);
    return fault_tag_;
}

std::string const &DeferralBridge::fault_text() const
{
    std::lock_guard<std::mutex> lock(mu_);
    return fault_text_;
}

bool DeferralBridge::idle() const
{
    std::lock_guard<std::mutex> lock(mu_);
    for (auto const &kv : lanes_)
    {
        std::lock_guard<std::mutex> lane_lock(kv.second->mu);
        if (!kv.second->steps.empty() || kv.second->executing ||
            kv.second->trailing)
        {
            return false;
        }
    }
    return true;
}

void DeferralBridge::serve_(LaneId lane, LaneState *st)
{
    // One lane's waiter: walks ITS queue in issue order. A gate parks
    // on the host lane's flag (the lane's own cv machinery); a
    // record-wait parks on the executed-record frontier (an earlier
    // issue position's mark, produced by the engine thread's direct
    // records or another lane's waiter); a call executes inline. A
    // refusal parks the lane (nothing after the faulting step
    // executes); the drain delivers the census, teardown releases
    // the park.
    (void)lane;
    for (;;)
    {
        Step step;
        {
            std::unique_lock<std::mutex> lock(st->mu);
            st->cv_work.wait(lock, [this, &st] {
                return stop_flag_.load(std::memory_order_acquire) ||
                    !st->steps.empty();
            });
            if (stop_flag_.load(std::memory_order_acquire))
            {
                return; // the destructor's join
            }
            step = std::move(st->steps.front());
            st->executing = true;
        }


        std::uint64_t const t0 = bridge_now_ns();
        std::uint64_t parked_ns = 0;
        bool surfaced = false;
        std::string fault_text;
        if (step.kind == Step::Kind::Gate)
        {
            gates_parked_.fetch_add(1, std::memory_order_relaxed);
            std::uint64_t const p0 = bridge_now_ns();
            bool const released =
                !host_->wait_flag(step.gate, stop_flag_);
            parked_ns = bridge_now_ns() - p0;
            if (released)
            {
                if (stop_flag_.load(std::memory_order_acquire))
                {
                    std::lock_guard<std::mutex> lock(st->mu);
                    st->executing = false;
                    return; // teardown, not a fault
                }
                surfaced = true;
                fault_text =
                    "a host-lane fault released a gate parked on its "
                    "mint flag (the producer never minted; the "
                    "fault-aware release)";
            }
        }
        else if (step.kind == Step::Kind::RecordWait)
        {
            // Park on the executed-record frontier, then EXECUTE the
            // step's body — the stream wait's enqueue — with the same
            // refusal handling a plain call has.
            record_waits_.fetch_add(1, std::memory_order_relaxed);
            std::uint64_t const p0 = bridge_now_ns();
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_recorded_.wait(lock, [this, &step] {
                    return stop_flag_.load(std::memory_order_acquire) ||
                        recorded_.count(step.mint) != 0 || any_faulted_ ||
                        host_faulted_;
                });
                if (recorded_.count(step.mint) == 0)
                {
                    if (stop_flag_.load(std::memory_order_acquire))
                    {
                        std::unique_lock<std::mutex> back(st->mu);
                        st->executing = false;
                        return; // teardown, not a fault
                    }
                    surfaced = true;
                    // The HOST lane's fault is the release (the
                    // deadlock closure): the awaited record fires on
                    // the host worker — its fault means the record
                    // never lands. faulted() is a join-time census
                    // fact, never an event query.
                    fault_text = host_faulted_
                        ? "a recorded-wait released by a HOST-lane "
                          "fault (the awaited mint's record twin never "
                          "fired)"
                        : "a recorded-wait released into the fault "
                          "path (a lane faulted while the awaited "
                          "mint's record stood queued)";
                }
            }
            parked_ns = bridge_now_ns() - p0;
            if (!surfaced)
            {
                try
                {
                    step.call(); // the stream wait, past the record
                }
                catch (std::exception const &ex)
                {
                    surfaced = true;
                    fault_text = std::string("the deferred enqueue was "
                                             "refused: ") +
                        ex.what();
                }
            }
        }
        else
        {
            try
            {
                step.call();
            }
            catch (std::exception const &ex)
            {
                surfaced = true;
                fault_text = std::string("the deferred enqueue was "
                                         "refused: ") +
                    ex.what();
            }
        }
        waiter_busy_ns_.fetch_add(
            bridge_now_ns() - t0 - parked_ns, std::memory_order_relaxed);
        waiter_parked_ns_.fetch_add(parked_ns, std::memory_order_relaxed);
        waiter_steps_.fetch_add(1, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lock(st->mu);
            st->executing = false;
            if (surfaced && !st->faulted)
            {
                st->faulted = true;
                st->fault_tag = step.tag;
                st->fault_text = std::move(fault_text);
            }
            if (!surfaced)
            {
                st->steps.pop_front();
            }
            st->cv_done.notify_all();
        }
        if (surfaced)
        {
            break; // the park below; nothing further runs on the lane
        }
    }

    // The fault park: held until teardown (the drain delivered the
    // census; the destructor's stop + kick releases it).
    std::unique_lock<std::mutex> lock(st->mu);
    st->cv_work.wait(lock,
        [this] { return stop_flag_.load(std::memory_order_acquire); });
}

} // namespace nnhaul
