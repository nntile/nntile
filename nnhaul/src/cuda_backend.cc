/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * CUDA backend's acts: the event provision's lease discipline, the
 * shared stream tail's ordering operations, the arena lane's addresses and
 * registration table, the two transfer channels' span-exact copies,
 * the host-mint gate, and the init-time selection, probe, and enable.
 *
 * @file src/cuda_backend.cc
 * The acts behind src/cuda_backend.hh: cudaEventRecord /
 * cudaStreamWaitEvent / cudaLaunchHostFunc / cudaStreamSynchronize
 * over one non-blocking stream per lane; the span-exact
 * cudaMemcpyAsync / cudaMemcpyPeerAsync (exactly the occupied span the
 * engine submitted); one cudaMalloc arena with stable chunk addresses;
 * per-(node, TaskId) registration; and the init-time helpers —
 * NNHAUL_DEVICES selection, the cudaDeviceCanAccessPeer probe (the
 * run's ONLY device introspection), and the post-context
 * cudaDeviceEnablePeerAccess. Every failure is a loud census, never
 * serviced; no event or lane state is ever read.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "cuda_backend.hh"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

namespace nnhaul
{

namespace
{

// The smallest power of two >= v (pow2ceil(0) == pow2ceil(1) == 1).
// Exact for v <= 2^63; the provision arithmetic's inputs are plan op
// counts, far below any wrap.
std::size_t pow2ceil(std::size_t v)
{
    if (v <= 1)
    {
        return 1;
    }
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

// A provision of zero is a configuration fault; validating before any
// resource exists keeps the failure clean (the host lane's same
// refusal).
std::size_t checked_provision(std::size_t n)
{
    if (n == 0)
    {
        throw std::runtime_error(
            "cuda event pool: a provision of zero is a configuration "
            "fault (sized as pow2ceil(2 * max_p m_p) "
            "floored at 1024, NNHAUL_EVENT_POOL overriding)");
    }
    return n;
}

// NNHAUL_EVENT_POOL as a positive handle count, 0 when unset; the
// knob OVERRIDES the sizing arithmetic. Read per
// call, so scenarios can set and unset it around inits.
std::size_t event_pool_override()
{
    char const *e = std::getenv("NNHAUL_EVENT_POOL");
    if (e == nullptr)
    {
        return 0;
    }
    char *end = nullptr;
    errno = 0;
    long const v = std::strtol(e, &end, 10);
    if (end == e || *end != '\0' || errno == ERANGE || v <= 0)
    {
        throw std::runtime_error(
            "NNHAUL_EVENT_POOL must be a positive event-handle count "
            "");
    }
    if (static_cast<unsigned long long>(v) >
        static_cast<unsigned long long>(
            std::numeric_limits<std::size_t>::max()))
    {
        throw std::runtime_error("NNHAUL_EVENT_POOL is impossibly large");
    }
    return static_cast<std::size_t>(v);
}

// NNHAUL_LIMIT_CUDA_MEM as bytes, 0 when unset; garbage is a loud
// init error (the same parse as engine.cc's
// limit_cuda_mem_bytes).
std::size_t limit_cuda_mem_bytes()
{
    char const *e = std::getenv("NNHAUL_LIMIT_CUDA_MEM");
    if (e == nullptr)
    {
        return 0;
    }
    char *end = nullptr;
    errno = 0;
    long mib = std::strtol(e, &end, 10);
    if (end == e || *end != '\0' || errno == ERANGE || mib <= 0)
    {
        throw std::runtime_error(
            "NNHAUL_LIMIT_CUDA_MEM must be a positive number of MiB");
    }
    // A long of MiB can exceed size_t once shifted; reject the absurd
    // rather than wrap the cap.
    if (static_cast<unsigned long>(mib) >
        (std::numeric_limits<std::size_t>::max() >> 20))
    {
        throw std::runtime_error("NNHAUL_LIMIT_CUDA_MEM is impossibly large");
    }
    return static_cast<std::size_t>(mib) << 20;
}

// The wait_host_flag gate's payload: the host worker's mint flag,
// boxed because cudaLaunchHostFunc carries one void*. The box is
// freed by the host function itself once it fires (or by the
// submission-refusing caller on a failed enqueue) — the one
// post-construction allocation, the host FIFO's own harness-fact
// discipline.
struct HostFlagGate
{
    std::atomic<std::uint32_t> *flag;
};

// The gate body: blocks its driver host-func thread on the flag —
// the cost of this mechanism — touching no
// CUDA API and reading no event state (the flag seam is the
// documented backend mechanism, host_worker.hh).
void host_flag_gate(void *arg)
{
    auto *gate = static_cast<HostFlagGate *>(arg);
    while (gate->flag->load(std::memory_order_acquire) == 0u)
    {
        std::this_thread::yield(); // blocked, not burning
    }
    delete gate;
}

} // namespace

// ---- CudaEventPool  ----

CudaEventPool::CudaEventPool(int ordinal, std::size_t provision)
    : ordinal_(ordinal),
      provision_(checked_provision(provision)),
      events_(provision_, cudaEvent_t{}),
      used_((provision_ + 63u) / 64u, 0u)
{
    cudaError_t err = cudaSetDevice(ordinal_);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("cuda event pool: ") +
            "cudaSetDevice(" + std::to_string(ordinal_) +
            "): " + cudaGetErrorString(err));
    }
    for (std::size_t i = 0; i < provision_; ++i)
    {
        // cudaEventDisableTiming — no timing is ever read; the
        // events exist to be recorded and waited, nothing else.
        err = cudaEventCreateWithFlags(&events_[i], cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                "cuda event pool: cudaEventCreateWithFlags(#" +
                std::to_string(i) + " of the provision " +
                std::to_string(provision_) + ") on device " +
                std::to_string(ordinal_) + " failed: " +
                cudaGetErrorString(err));
        }
    }
}

CudaEventPool::~CudaEventPool()
{
    // Destruction belongs after the barriers; failures are
    // diagnostics, a destructor never throws.
    (void)cudaSetDevice(ordinal_);
    for (std::size_t i = 0; i < provision_; ++i)
    {
        if (cudaEventDestroy(events_[i]) != cudaSuccess)
        {
            std::fprintf(stderr,
                "[nnhaul] cuda event pool: cudaEventDestroy(#%zu of "
                "device %d) failed at shutdown\n",
                i, ordinal_);
        }
    }
}

void CudaEventPool::extend(std::size_t new_provision)
{
    if (new_provision <= provision_)
    {
        return; // the provision is a monotone floor
    }
    // The new events are created into a TEMP first: a failure mid-way
    // destroys them and leaves the pool exactly as it was — provision_,
    // events_, and the leased bitmap all untouched — so a retry of the
    // same extend re-creates the full additional set (nothing
    // half-committed, no null event ever leasable).
    // Events belong to the pool's device context, not to whatever
    // device happened to be current when the plan grew the provision.
    int previous_device = -1;
    if(cudaGetDevice(&previous_device) != cudaSuccess)
    {
        throw std::runtime_error(
            "cuda event pool: extend cannot read the current device");
    }
    if(cudaSetDevice(ordinal_) != cudaSuccess)
    {
        throw std::runtime_error(
            "cuda event pool: extend cannot bind device " +
            std::to_string(ordinal_));
    }
    std::vector<cudaEvent_t> fresh(new_provision - provision_);
    for (std::size_t i = 0; i < fresh.size(); ++i)
    {
        cudaError_t err =
            cudaEventCreateWithFlags(&fresh[i], cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            for (cudaEvent_t e : fresh)
            {
                if (e != nullptr)
                {
                    cudaEventDestroy(e);
                }
            }
            if(previous_device >= 0)
            {
                (void) cudaSetDevice(previous_device);
            }
            throw std::runtime_error(
                "cuda event pool: extend's cudaEventCreateWithFlags(#" +
                std::to_string(provision_ + i) + ") on device " +
                std::to_string(ordinal_) + " failed: " +
                cudaGetErrorString(err));
        }
    }
    if(previous_device >= 0)
    {
        (void) cudaSetDevice(previous_device);
    }
    std::lock_guard<std::mutex> lock(events_mu_);
    // The leased bitmap RESIZES, never resets: leases from plans still
    // in flight (the pipelined depth-2 shape — a plan's mints live in
    // the registry until ITS close sweeps them) must survive the
    // growth commit. The old words keep their bits (acquire never sets
    // a bit beyond the provision it knew); the new words start empty.
    events_.reserve(new_provision);
    provision_ = new_provision;
    for (cudaEvent_t e : fresh)
    {
        events_.push_back(e);
    }
    used_.resize((provision_ + 63u) / 64u, 0u);
}

EventHandle CudaEventPool::acquire()
{
    for (std::size_t w = 0; w < used_.size(); ++w)
    {
        std::size_t const wi = (hint_ + w) % used_.size();
        // The word's VALID bits: the provision's tail word carries
        // bits past the provision that no handle names.
        std::uint64_t const valid =
            (wi + 1u == used_.size() && provision_ % 64u != 0u)
                ? ((std::uint64_t{1} << (provision_ % 64u)) - 1u)
                : ~std::uint64_t{0};
        std::uint64_t const free_bits = ~used_[wi] & valid;
        if (free_bits == 0)
        {
            continue;
        }
        std::size_t b = 0;
        while ((free_bits & (std::uint64_t{1} << b)) == 0)
        {
            ++b;
        }
        used_[wi] |= std::uint64_t{1} << b;
        ++used_count_;
        hint_ = wi;
        return static_cast<EventHandle>(wi * 64u + b);
    }
    // Exhaustion is a configuration fault, loud with a census naming
    // the provision and how it is sized.
    throw std::runtime_error(
        "cuda event pool (device " + std::to_string(ordinal_) +
        ") exhausted: configuration fault — provision " +
        std::to_string(provision_) + " handles, " +
        std::to_string(used_count_) + " leased; the provision is sized "
        "as pow2ceil(2 * max_p m_p) floored at 1024, with "
        "NNHAUL_EVENT_POOL overriding — a deployment whose queues can "
        "reach the displacement ceiling (outstanding <= C_total * (1 + "
        "L)) must raise NNHAUL_EVENT_POOL to it at init");
}

void CudaEventPool::release(EventHandle e)
{
    std::size_t const at = static_cast<std::size_t>(e);
    if (at >= provision_)
    {
        throw std::runtime_error(
            "cuda event pool (device " + std::to_string(ordinal_) +
            "): release of event " + std::to_string(e) +
            " outside the provision " + std::to_string(provision_) +
            " — a bookkeeping defect");
    }
    std::uint64_t const bit = std::uint64_t{1} << (at % 64u);
    if ((used_[at / 64u] & bit) == 0)
    {
        throw std::runtime_error(
            "cuda event pool (device " + std::to_string(ordinal_) +
            "): release of the unleased event " + std::to_string(e) +
            " — a bookkeeping defect (a double release)");
    }
    used_[at / 64u] &= ~bit;
    --used_count_;
}

cudaEvent_t CudaEventPool::raw(EventHandle e) const
{
    // Under events_mu_: the host worker's record twin reads here
    // cross-thread, and extend() (the engine's growth commit) appends
    // to events_ under the same guard.
    std::lock_guard<std::mutex> lock(events_mu_);
    if (static_cast<std::size_t>(e) >= provision_)
    {
        // A submission error refused at enqueue, never a silent skip.
        throw std::runtime_error(
            "cuda event pool (device " + std::to_string(ordinal_) +
            "): event " + std::to_string(e) + " outside the provision " +
            std::to_string(provision_) + " — refused at submission");
    }
    return events_[e];
}

std::size_t CudaEventPool::suggest_event_provision(
    std::vector<std::size_t> const &m_p)
{
    std::size_t const override_count = event_pool_override();
    if (override_count != 0)
    {
        return override_count;
    }
    std::size_t max_m = 0;
    for (std::size_t v : m_p)
    {
        max_m = std::max(max_m, v);
    }
    // Depth-2 pipelining's typical outstanding set is
    // mints(p) + mints(p+1) <= 2 * max_p m_p; pow2ceil of it, floored
    // at 1024.
    std::size_t const twice =
        (max_m > std::numeric_limits<std::size_t>::max() / 2u)
            ? std::numeric_limits<std::size_t>::max()
            : 2u * max_m;
    return std::max<std::size_t>(1024u, pow2ceil(twice));
}

// ---- CudaLaneOps  ----

CudaLaneOps::CudaLaneOps(int ordinal, CudaEventPool &events)
    : ordinal_(ordinal), events_(&events)
{
    // The engine's init made the contexts current;
    // this binding is idempotent and keeps the lane self-contained.
    cudaError_t err = cudaSetDevice(ordinal_);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("cuda backend: ") +
            "cudaSetDevice(" + std::to_string(ordinal_) +
            "): " + cudaGetErrorString(err));
    }
    // One non-blocking stream per lane, created in
    // that device's context.
    err = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cuda backend: cudaStreamCreateWithFlags("
            "cudaStreamNonBlocking) on device " +
            std::to_string(ordinal_) + " failed: " +
            cudaGetErrorString(err));
    }
}

CudaLaneOps::~CudaLaneOps()
{
    // The engine's shutdown barriers precede destruction; a failure is a
    // diagnostic, a destructor never throws.
    (void)cudaSetDevice(ordinal_);
    if (stream_ != nullptr && cudaStreamDestroy(stream_) != cudaSuccess)
    {
        std::fprintf(stderr,
            "[nnhaul] cuda backend: cudaStreamDestroy on device %d "
            "failed at shutdown\n",
            ordinal_);
    }
}

void CudaLaneOps::mint(EventHandle e)
{
    cudaError_t err = cudaEventRecord(events_->raw(e), stream_);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cuda backend: cudaEventRecord on device " +
            std::to_string(ordinal_) + " failed: " +
            cudaGetErrorString(err) + " — refused at submission");
    }
}

void CudaLaneOps::wait(EventHandle e)
{
    // CUDA's wait captures the event's state AT CALL TIME; the engine
    // guarantees the record call strictly precedes it.
    cudaError_t err = cudaStreamWaitEvent(stream_, events_->raw(e), 0);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cuda backend: cudaStreamWaitEvent on device " +
            std::to_string(ordinal_) + " failed: " +
            cudaGetErrorString(err) + " — refused at submission");
    }
}

void CudaLaneOps::wait_from(EventPool &pool, EventHandle e)
{
    // Resolve the handle in its HOME pool, not this lane's. A peer COPY
    // waits the source device's mint; resolving pool-relative indices
    // against the consumer's pool would wait an unrelated event.
    auto *cuda_pool = dynamic_cast<CudaEventPool *>(&pool);
    if (cuda_pool == nullptr)
    {
        wait(e);
        return;
    }
    cudaError_t err = cudaStreamWaitEvent(stream_, cuda_pool->raw(e), 0);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cuda backend: cross-pool cudaStreamWaitEvent on device " +
            std::to_string(ordinal_) + " failed: " +
            cudaGetErrorString(err) + " — refused at submission");
    }
}

void CudaLaneOps::notify(NotifyHook fn, void *arg)
{
    if (fn == nullptr)
    {
        throw std::runtime_error(
            "cuda backend: notify of a null hook on device " +
            std::to_string(ordinal_) + " refused at submission");
    }
    // The once-executed host callback at this lane position —
    // after earlier-enqueued work, ordering later-enqueued work. The
    // hook carries no CUDA API of its own and its bounded work is
    // the caller's discipline (the host-flag gates ride this
    // mechanism).
    cudaError_t err = cudaLaunchHostFunc(stream_, fn, arg);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cuda backend: cudaLaunchHostFunc on device " +
            std::to_string(ordinal_) + " failed: " +
            cudaGetErrorString(err) + " — refused at submission");
    }
}

void CudaLaneOps::barrier()
{
    // The lane's whole tail joined — queue close and run end
    // only. A faulted op's async failure is delivered HERE, the
    // detection point; the ENGINE wraps the census with the op's own
    // (the interface seam — the lane cannot know the OpRef). The
    // per-lane deferral bridge joins
    // through the EXECUTOR's light close — per-lane drain +
    // terminal-event join — never through this whole-stream sync;
    // this path stands for teardown and forced joins.
    cudaError_t err = cudaStreamSynchronize(stream_);
    if (err == cudaSuccess)
    {
        // A context-level sticky fault (an op that faulted on ANOTHER
        // lane) leaves this stream's sync clean while later enqueues
        // on every lane silently refuse — the CI's container flake
        // read stale bytes through exactly this hole. The join is
        // the detection point for the whole context, not only this
        // stream: deliver the sticky fault here, never at teardown.
        err = cudaGetLastError(); // also clears it
    }
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cuda stream barrier on device " + std::to_string(ordinal_) +
            ": " + cudaGetErrorString(err) +
            " — a faulted op's failure is delivered at the join; "
            "the abort precedes further issue and a fault is never "
            "serviced");
    }
}

void CudaEventPool::join(EventHandle e)
{
    // EventPool::join's CUDA half (the batch-granular barrier): block
    // the engine thread until THIS event completed on its stream —
    // the successor's fed work sits behind it and never delays the
    // join. The sticky async error, if any, is delivered as the
    // throw's text.
    cudaError_t const err = cudaEventSynchronize(raw(e));
    if (err != cudaSuccess)
    {
        cudaGetLastError(); // clear the sticky diagnostic
        throw std::runtime_error(
            "cuda event barrier on device " +
            std::to_string(ordinal_) + ": " + cudaGetErrorString(err) +
            " — a faulted op's failure is delivered at the join; "
            "the abort precedes further issue and a fault is never "
            "serviced");
    }
}

void CudaLaneOps::wait_host_flag(
    HostWorkerLane &host, EventHandle host_e)
{
    std::atomic<std::uint32_t> *flag = host.flag(host_e); // loud on range
    auto *gate = new HostFlagGate{flag};
    cudaError_t err = cudaLaunchHostFunc(stream_, host_flag_gate, gate);
    if (err != cudaSuccess)
    {
        delete gate;
        throw std::runtime_error(
            "cuda backend: the host-mint gate's cudaLaunchHostFunc on "
            "device " + std::to_string(ordinal_) + " failed: " +
            cudaGetErrorString(err) + " — refused at submission");
    }
}

// ---- the stamp mechanism, the device-lane realization ----

EventHandle CudaLaneOps::stamp_acquire()
{
    std::lock_guard<std::mutex> guard(stamp_mu_);
    if (stamp_free_.empty())
    {
        // The growth batch: the provision grows to the working set
        // (the un-retired plans' stamps — one clock-read slot per
        // instrumented op, plus the plan's tail stamp per touched
        // lane) and recycles at the harvests, so growth settles at
        // the largest working set and never grows again. A reserve
        // failure is the loud configuration fault, never serviced — and
        // the allocation is plain
        // host memory: no CUDA call, no device tax.
        constexpr std::size_t kBatch = 256;
        std::size_t const base = stamp_values_.size();
        try
        {
            stamp_values_.resize(base + kBatch, 0);
            stamp_leased_.resize(base + kBatch, true);
        }
        catch (std::bad_alloc const &)
        {
            throw std::runtime_error(
                "cuda backend (device " + std::to_string(ordinal_) +
                "): the stamp provision's growth to " +
                std::to_string(base + kBatch) +
                " slots failed — a configuration fault, never "
                "serviced");
        }
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

void CudaLaneOps::stamp_release(EventHandle h)
{
    std::lock_guard<std::mutex> guard(stamp_mu_);
    std::size_t const at = static_cast<std::size_t>(h);
    if (at >= stamp_values_.size())
    {
        throw std::runtime_error(
            "cuda backend (device " + std::to_string(ordinal_) +
            "): stamp_release of stamp " + std::to_string(h) +
            " outside the provision " +
            std::to_string(stamp_values_.size()) +
            " — a bookkeeping defect");
    }
    if (!stamp_leased_[at])
    {
        throw std::runtime_error(
            "cuda backend (device " + std::to_string(ordinal_) +
            "): stamp_release of the unleased stamp " +
            std::to_string(h) + " — a bookkeeping defect (a double "
            "release)");
    }
    stamp_leased_[at] = false;
    stamp_values_[at] = 0;
    stamp_free_.push_back(h);
}

void CudaLaneOps::stamp(EventHandle h)
{
    std::lock_guard<std::mutex> guard(stamp_mu_);
    // The clock read AT this lane position — the submission's
    // instant, on the submitting thread (the engine thread direct,
    // or the bridge's waiter in the same per-lane order). No CUDA
    // API: the read taxes nothing (the cost rationale in the header).
    std::size_t const at = static_cast<std::size_t>(h);
    if (at >= stamp_values_.size() || !stamp_leased_[at])
    {
        throw std::runtime_error(
            "cuda backend (device " + std::to_string(ordinal_) +
            "): stamp record of stamp " + std::to_string(h) +
            " outside the leased provision — refused at submission");
    }
    stamp_values_[at] =
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
}

StampFact CudaLaneOps::stamp_read(EventHandle before, EventHandle after)
{
    std::lock_guard<std::mutex> guard(stamp_mu_);
    // The queue-close join's read: the barriers
    // retired every recorded position, so both slots carry their
    // positions' clock reads. A slot of zero is unwritten — the
    // barrier's own defect, loud (the join is the detection point),
    // never serviced.
    std::size_t const b = static_cast<std::size_t>(before);
    std::size_t const a = static_cast<std::size_t>(after);
    if (b >= stamp_values_.size() || a >= stamp_values_.size())
    {
        throw std::runtime_error(
            "cuda backend (device " + std::to_string(ordinal_) +
            "): stamp read of a handle outside the provision — a "
            "bookkeeping defect");
    }
    if (stamp_values_[b] == 0 || stamp_values_[a] == 0)
    {
        throw std::runtime_error(
            "cuda backend (device " + std::to_string(ordinal_) +
            "): a stamp slot of the pair is unwritten at the "
            "queue-close join — the barriers did not retire the "
            "position, a barrier defect");
    }
    StampFact fact;
    fact.start_ns = stamp_values_[b];
    fact.end_ns = stamp_values_[a];
    return fact;
}

// ---- CudaExecLane  ----

CudaExecLane::CudaExecLane(int ordinal, std::size_t limit_chunks,
    std::size_t initial, CudaEventPool &events)
    : ops_(ordinal, events), limit_(limit_chunks)
{
    // No preallocated memory by default;
    // the arena starts at the explicit preallocation ask, CHECKED
    // against the soft limit (a head start above the limit is a loud
    // configuration fault). An init-time ask the device cannot back
    // is loud the same way.
    cudaError_t err = cudaSetDevice(ops_.ordinal());
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("cuda backend: ") +
            "cudaSetDevice(" + std::to_string(ops_.ordinal()) +
            "): " + cudaGetErrorString(err));
    }
    if (limit_ != 0 && initial > limit_)
    {
        throw std::runtime_error(
            "cuda exec lane (device " + std::to_string(ordinal) +
            "): the explicit preallocation of " +
            std::to_string(initial) +
            " chunks exceeds the soft limit of " +
            std::to_string(limit_) +
            " chunks — raise the limit or lower the preallocation");
    }
    std::size_t const ask = initial;
    if (ask > 0 && grow(ask) < ask) // partial delivery is a fault too

    {
        throw std::runtime_error(
            "cuda exec lane (device " + std::to_string(ordinal) +
            "): the explicit preallocation of " + std::to_string(ask) +
            " chunks failed (the device backed " +
            std::to_string(chunks_.size()) + ")");
    }
}

CudaExecLane::~CudaExecLane()
{
    // The engine's shutdown barriers precede destruction;
    // failures are diagnostics, a destructor never throws.
    (void)cudaSetDevice(ops_.ordinal());
    // Segments free as segments (a chunk is interior).
    for (Segment &seg : segments_)
    {
        if (cudaFree(seg.base) != cudaSuccess)
        {
            std::fprintf(stderr,
                "[nnhaul] cuda exec lane: a %zu-chunk segment cudaFree "
                "on device %d failed at shutdown\n",
                seg.chunks, ops_.ordinal());
        }
    }
}

// The allocation ladder for ONE event of
// `chunks` chunks: TILE-WISE first; SINGLE chunks for this event only
// on error; a failed single-chunk allocation flips the pool
// NON-ALLOCABLE, permanently. The soft limit bounds every event.
std::size_t CudaExecLane::alloc_event(std::size_t chunks)
{
    if (chunks == 0 || hard_capped_)
    {
        return 0;
    }
    std::size_t remaining = limit_ != 0 ? limit_ - chunks_.size()
                                        : std::numeric_limits<
                                              std::size_t>::max();
    if (remaining == 0)
    {
        return 0; // the soft wall: the budget is fixed
    }
    std::size_t const want = std::min(chunks, remaining);
    if (allocate_segment(want))
    {
        return want; // tile-wise landed whole
    }
    if (want <= 1)
    {
        enter_non_allocable();
        return 0;
    }
    std::size_t added = 0;
    while (added < want)
    {
        remaining = limit_ != 0 ? limit_ - chunks_.size()
                                : std::numeric_limits<
                                      std::size_t>::max();
        if (remaining == 0)
        {
            break; // the soft wall mid-event
        }
        if (!allocate_segment(1))
        {
            enter_non_allocable(); // not even one chunk: allocations off
            break;
        }
        added += 1;
    }
    return added;
}

std::size_t CudaExecLane::grow_to_wall(std::size_t tile_chunks)
{
    // The planning-time growth: the tile
    // through the ladder; under a soft limit, the remaining budget as
    // one more event (grow is possible up to the limit); an unlimited
    // pool serves the demand only. Nothing freed until the lane's
    // death.
    if (tile_chunks == 0 || hard_capped_)
    {
        return 0;
    }
    std::size_t const before = chunks_.size();
    (void) alloc_event(tile_chunks);
    if (limit_ != 0 && !hard_capped_ && chunks_.size() < limit_)
    {
        (void) alloc_event(limit_ - chunks_.size());
    }
    return chunks_.size() - before;
}

std::size_t CudaExecLane::grow(std::size_t n)
{
    // ENSURE-exactly-n for the init-time preallocation ask: the ladder.
    if (chunks_.size() >= n)
    {
        return 0;
    }
    return alloc_event(n - chunks_.size());
}

// One allocation attempt of exactly `seg_chunks` chunks; false
// WITHOUT the mode decision (the ladder owns it).
bool CudaExecLane::allocate_segment(std::size_t seg_chunks)
{
    if (seg_chunks == 0)
    {
        return false;
    }
    std::size_t const bytes = seg_chunks * chunk_class_bytes();
    void *seg = nullptr;
    if (cudaMalloc(&seg, bytes) != cudaSuccess || seg == nullptr)
    {
        return false;
    }
    segments_.push_back(Segment{seg, seg_chunks});
    chunks_.reserve(chunks_.size() + seg_chunks);
    for (std::size_t c = 0; c < seg_chunks; ++c)
    {
        chunks_.push_back(static_cast<char *>(seg) + c * chunk_class_bytes());
    }
    return true;
}

void CudaExecLane::enter_non_allocable()
{
    if (!hard_capped_)
    {
        hard_capped_ = true;
        std::fprintf(stderr,
            "[nnhaul] cuda exec lane (device %d): the growth episode "
            "hit the hard limit; growth closed at %zu chunks — "
            "we stick to what we have\n",
            ops_.ordinal(), chunks_.size());
    }
}

std::size_t CudaExecLane::contig_bytes(ChunkId chunk) const
{
    std::size_t base = 0;
    for (Segment const &seg : segments_)
    {
        if (static_cast<std::size_t>(chunk) < base + seg.chunks)
        {
            return (base + seg.chunks -
                       static_cast<std::size_t>(chunk)) *
                chunk_class_bytes();
        }
        base += seg.chunks;
    }
    return 0; // not held
}

void *CudaExecLane::addr(ChunkId chunk) const
{
    if (static_cast<std::size_t>(chunk) >= chunks_.size())
    {
        throw std::runtime_error(
            "cuda exec lane (device " + std::to_string(ops_.ordinal()) +
            "): chunk " + std::to_string(chunk) +
            " outside the pool cardinality " +
            std::to_string(chunks_.size()) + " — a defect");
    }
    // Stable for the run's life: a chunk never moves, splits, or
    // reallocates once allocated.
    return chunks_[static_cast<std::size_t>(chunk)];
}

void CudaExecLane::register_task(TaskId task, CudaLaunchFn fn)
{
    if (!fn)
    {
        throw std::runtime_error(
            "cuda exec lane (device " + std::to_string(ops_.ordinal()) +
            "): registration of task " + std::to_string(task) +
            " with a null body refused — one symbol, one body");
    }
    auto [it, inserted] = tasks_.emplace(task, std::move(fn));
    if (!inserted)
    {
        throw std::runtime_error(
            "cuda exec lane (device " + std::to_string(ops_.ordinal()) +
            "): task " + std::to_string(task) +
            " registered twice — one symbol, one body, resolved at "
            "INIT ");
    }
}

void CudaExecLane::launch(
    TaskId task, NodeId node, void *const *operand_addrs)
{
    (void)node; // this lane's own device — the engine's lane_of_exec
                // routed the TILE_OP here ; the node id
                // carries no further fact for the device lane.
    auto it = tasks_.find(task);
    if (it == tasks_.end())
    {
        // The enqueue detection point: a loud census-style refusal,
        // never a silent skip. The checker closes the class at check
        // time; this refusal is the residual backstop.
        throw std::runtime_error(
            "cuda exec lane (device " + std::to_string(ops_.ordinal()) +
            "): launch of task " + std::to_string(task) +
            " refused at submission — no body registered on this "
            "node (task identity resolves at INIT registration)");
    }
    // The registered body enqueues on the lane's stream and returns;
    // operand addresses are host-resolved before the call and
    // the array is the engine's to keep stable until served.
    it->second(ops_.stream(), operand_addrs);
}


std::size_t CudaExecLane::suggest_limit_bytes(int ordinal,
    std::size_t api_cap_bytes, std::size_t)
{
    // The cap precedence as the SOFT CEILING on demand growth — no
    // bytes are allocated here, the arena grows per chunk until this
    // limit or the hard limit:
    //   (1) NNHAUL_LIMIT_CUDA_MEM (wins absolutely, throws on bad
    //       values);
    //   (2) else a positive init argument — honored exactly;
    //   (3) else 0 — unlimited: growth ends at the first cudaMalloc
    //       failure only. Nothing is pre-carved at init.
    std::size_t cap = 0;
    std::size_t const env_cap = limit_cuda_mem_bytes(); // throws on bad
    if (env_cap != 0)
    {
        cap = env_cap;
    }
    else if (api_cap_bytes != 0)
    {
        cap = api_cap_bytes;
    }
    if (cap != 0 && cap < chunk_class_bytes())
    {
        // A sub-class cap backs zero chunks — every tile costs a
        // whole chunk_class_bytes() chunk.
        throw std::runtime_error(
            "device arena limit of " + std::to_string(cap) +
            " bytes on device " + std::to_string(ordinal) +
            " is below the one chunk class of " +
            std::to_string(chunk_class_bytes()) +
            " bytes — a sub-class limit is a loud init error "
            "");
    }
    return cap;
}

// ---- CudaHostChannel  ----

CudaHostChannel::CudaHostChannel(
    int ordinal, HostChannelDirection dir, CudaEventPool &events)
    : ops_(ordinal, events), dir_(dir)
{
}

CudaHostChannel::~CudaHostChannel() = default;

void CudaHostChannel::copy(
    void const *src, void *dst, std::size_t nbytes)
{
    // EXACTLY `nbytes` — the occupied span the engine submitted
    // (the whole class for a full chunk, the tail
    // chunk's payload remainder and nothing more); the host endpoint
    // is the host pool's pinned memory (DMA-stable), never a backend
    // staging buffer. The span's range is
    // checked and loud, never clipped.
    require_copyable_span(nbytes);
    cudaError_t err = cudaMemcpyAsync(dst, src, nbytes,
        dir_ == HostChannelDirection::HostToDevice
            ? cudaMemcpyHostToDevice
            : cudaMemcpyDeviceToHost,
        ops_.stream());
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cuda host channel (device " +
            std::to_string(ops_.ordinal()) + ", " +
            (dir_ == HostChannelDirection::HostToDevice ? "host->device"
                                                        : "device->host") +
            "): cudaMemcpyAsync of " + std::to_string(nbytes) +
            " bytes failed: " + cudaGetErrorString(err) +
            " — refused at submission");
    }
}

void CudaHostChannel::copy_run(
    void const *src, void *dst, std::size_t nbytes)
{
    // The merged-run realization: the byte count is the
    // SUM of the run's consecutive occupied spans — an interior span
    // whole-class, at most the last one a tail remainder — so the
    // count legitimately exceeds chunk_class_bytes() (copy()'s per-chunk
    // guard does not bind here; the executor proved the address
    // contiguity that makes the sum the exact same bytes over the same
    // addresses in the same lane order). Zero bytes is a fault, loud.
    if (nbytes == 0)
    {
        throw std::runtime_error(
            "cuda host channel (device " +
            std::to_string(ops_.ordinal()) + ", " +
            (dir_ == HostChannelDirection::HostToDevice ? "host->device"
                                                        : "device->host") +
            "): copy_run of zero bytes refused — a merged COPY run moves "
            "at least one occupied span ");
    }
    cudaError_t err = cudaMemcpyAsync(dst, src, nbytes,
        dir_ == HostChannelDirection::HostToDevice
            ? cudaMemcpyHostToDevice
            : cudaMemcpyDeviceToHost,
        ops_.stream());
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cuda host channel (device " +
            std::to_string(ops_.ordinal()) + ", " +
            (dir_ == HostChannelDirection::HostToDevice ? "host->device"
                                                        : "device->host") +
            "): cudaMemcpyAsync of " + std::to_string(nbytes) +
            " bytes (a merged COPY run) failed: " +
            cudaGetErrorString(err) + " — refused at submission");
    }
}

// ---- CudaPeerChannel  ----

CudaPeerChannel::CudaPeerChannel(
    int src_ordinal, int dst_ordinal, CudaEventPool &events)
    : ops_(dst_ordinal, events), // the stream belongs to the pair's
                                 // DESTINATION device
      src_ordinal_(src_ordinal)
{
    if (src_ordinal == dst_ordinal)
    {
        throw std::runtime_error(
            "cuda peer channel: a self-pair (device " +
            std::to_string(src_ordinal) + " to itself) refused — a "
            "channel needs two distinct endpoints");
    }
}

CudaPeerChannel::~CudaPeerChannel() = default;

void CudaPeerChannel::copy(
    void const *src, void *dst, std::size_t nbytes)
{
    // EXACTLY `nbytes` — the occupied span the engine submitted,
    // over the probed, peer-enabled pair,
    // on the destination device's stream. The span's
    // range is checked and loud, never clipped.
    require_copyable_span(nbytes);
    cudaError_t err = cudaMemcpyPeerAsync(dst, ops_.ordinal(), src,
        src_ordinal_, nbytes, ops_.stream());
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cuda peer channel (device " +
            std::to_string(src_ordinal_) + " -> device " +
            std::to_string(ops_.ordinal()) +
            "): cudaMemcpyPeerAsync of " + std::to_string(nbytes) +
            " bytes failed: " + cudaGetErrorString(err) +
            " — refused at submission");
    }
}

void CudaPeerChannel::copy_run(
    void const *src, void *dst, std::size_t nbytes)
{
    // The merged-run realization over the probed pair:
    // the run's SUM of consecutive occupied spans (interior spans
    // whole-class, at most the last a tail remainder) in ONE
    // cudaMemcpyPeerAsync on the destination device's stream. Zero
    // bytes is a fault, loud.
    if (nbytes == 0)
    {
        throw std::runtime_error(
            "cuda peer channel (device " +
            std::to_string(src_ordinal_) + " -> device " +
            std::to_string(ops_.ordinal()) +
            "): copy_run of zero bytes refused — a merged COPY run moves "
            "at least one occupied span ");
    }
    cudaError_t err = cudaMemcpyPeerAsync(dst, ops_.ordinal(), src,
        src_ordinal_, nbytes, ops_.stream());
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cuda peer channel (device " +
            std::to_string(src_ordinal_) + " -> device " +
            std::to_string(ops_.ordinal()) +
            "): cudaMemcpyPeerAsync of " + std::to_string(nbytes) +
            " bytes (a merged COPY run) failed: " +
            cudaGetErrorString(err) + " — refused at submission");
    }
}

// ---- the init-time helpers  ----

std::vector<int> select_devices(int ncuda)
{
    if (ncuda < 0)
    {
        throw std::runtime_error(
            "select_devices: ncuda of " + std::to_string(ncuda) +
            " is negative — a loud init error ");
    }
    std::vector<int> ordinals;
    char const *e = std::getenv("NNHAUL_DEVICES");
    if (e == nullptr)
    {
        // The default: ordinals 0..ncuda-1 ascending.
        ordinals.reserve(static_cast<std::size_t>(ncuda));
        for (int i = 0; i < ncuda; ++i)
        {
            ordinals.push_back(i);
        }
        return ordinals;
    }
    std::string const listing = e;
    if (listing.empty())
    {
        throw std::runtime_error(
            "NNHAUL_DEVICES is set but empty — expected comma-separated "
            "CUDA ordinals ");
    }
    std::size_t at = 0;
    while (at <= listing.size())
    {
        std::size_t const comma = listing.find(',', at);
        std::string const token =
            listing.substr(at, comma == std::string::npos
                                   ? std::string::npos
                                   : comma - at);
        if (token.empty() ||
            token.find_first_not_of("0123456789") != std::string::npos)
        {
            throw std::runtime_error(
                "NNHAUL_DEVICES token '" + token +
                "' is not a CUDA ordinal — expected comma-separated "
                "non-negative integers ");
        }
        errno = 0;
        char *end = nullptr;
        long const v = std::strtol(token.c_str(), &end, 10);
        if (errno == ERANGE ||
            v > static_cast<long>(std::numeric_limits<int>::max()))
        {
            throw std::runtime_error(
                "NNHAUL_DEVICES token '" + token +
                "' is an impossibly large ordinal ");
        }
        ordinals.push_back(static_cast<int>(v));
        at = (comma == std::string::npos) ? listing.size() + 1
                                          : comma + 1;
    }
    if (ordinals.size() != static_cast<std::size_t>(ncuda))
    {
        // The knob selects WHICH k devices, ncuda says HOW MANY; they
        // MUST agree.
        throw std::runtime_error(
            "NNHAUL_DEVICES lists " + std::to_string(ordinals.size()) +
            " ordinals (" + listing + ") but ncuda is " +
            std::to_string(ncuda) +
            " — the knob selects WHICH k devices, ncuda says HOW "
            "MANY; they MUST agree ");
    }
    // Device node i IS the i-th smallest selected
    // ordinal, whatever order the knob listed them in — enumerate
    // ascending, one NodeId per ordinal.
    std::sort(ordinals.begin(), ordinals.end());
    for (std::size_t i = 1; i < ordinals.size(); ++i)
    {
        if (ordinals[i - 1] == ordinals[i])
        {
            throw std::runtime_error(
                "NNHAUL_DEVICES lists ordinal " +
                std::to_string(ordinals[i]) +
                " twice — one NodeId per ordinal ");
        }
    }
    return ordinals;
}

std::vector<std::uint8_t> probe_peer_matrix(
    std::vector<int> const &ordinals)
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("cuda backend: cudaGetDeviceCount failed: ") +
            cudaGetErrorString(err));
    }
    for (int ordinal : ordinals)
    {
        if (ordinal < 0 || ordinal >= count)
        {
            throw std::runtime_error(
                "device ordinal " + std::to_string(ordinal) +
                " outside the driver's device count of " +
                std::to_string(count) +
                " — the selection cannot name it ");
        }
    }
    std::size_t const k = ordinals.size();
    std::vector<std::uint8_t> bits(k * k, 0u);
    // The run's ONLY device introspection : one probe
    // per ordered pair, the answer recorded FAITHFULLY — a pair may
    // be unidirectional, never mirrored; the diagonal is no channel.
    for (std::size_t i = 0; i < k; ++i)
    {
        for (std::size_t j = 0; j < k; ++j)
        {
            if (i == j)
            {
                continue;
            }
            int can = 0;
            err = cudaDeviceCanAccessPeer(
                &can, ordinals[i], ordinals[j]);
            if (err != cudaSuccess)
            {
                throw std::runtime_error(
                    "cuda backend: cudaDeviceCanAccessPeer(source " +
                    std::to_string(ordinals[i]) + ", destination " +
                    std::to_string(ordinals[j]) +
                    ") failed: " + cudaGetErrorString(err));
            }
            bits[i * k + j] = (can != 0) ? 1u : 0u;
        }
    }
    return bits;
}

void enable_peer_access(std::vector<int> const &ordinals,
    std::vector<std::uint8_t> const &peer_bits)
{
    std::size_t const k = ordinals.size();
    if (peer_bits.size() != k * k)
    {
        throw std::runtime_error(
            "enable_peer_access: a peer matrix is k*k row-major bits "
            "(probe_peer_matrix's shape) — got " +
            std::to_string(peer_bits.size()) + " bits for k = " +
            std::to_string(k) + " — a caller defect");
    }
    // AFTER the contexts: the caller made each
    // device's primary context current, and
    // cudaDeviceEnablePeerAccess requires the ACCESSING (source)
    // device's primary context current — exactly why enable cannot
    // sit inside the probe (no context exists at probe time).
    for (std::size_t i = 0; i < k; ++i)
    {
        for (std::size_t j = 0; j < k; ++j)
        {
            if (i == j || peer_bits[i * k + j] == 0u)
            {
                continue;
            }
            cudaError_t err = cudaSetDevice(ordinals[i]);
            if (err != cudaSuccess)
            {
                throw std::runtime_error(std::string("cuda backend: ") +
                    "cudaSetDevice(" + std::to_string(ordinals[i]) +
                    "): " + cudaGetErrorString(err));
            }
            err = cudaDeviceEnablePeerAccess(ordinals[j], 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled)
            {
                // Success-shaped: the pair is enabled already.
                cudaGetLastError();
                continue;
            }
            if (err != cudaSuccess)
            {
                throw std::runtime_error(
                    "cuda backend: cudaDeviceEnablePeerAccess(source " +
                    std::to_string(ordinals[i]) + " -> destination " +
                    std::to_string(ordinals[j]) +
                    ") failed: " + cudaGetErrorString(err));
            }
        }
    }
}

} // namespace nnhaul
