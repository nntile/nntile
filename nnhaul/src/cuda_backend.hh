/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * CUDA backend: CudaExecLane, CudaHostChannel, CudaPeerChannel,
 * CudaEventPool — plus the init-time device selection, peer probe, and
 * peer-enable helpers.
 *
 * @file src/cuda_backend.hh
 * The vendor home: vendor specifics live in this file and
 * src/cuda_backend.cc ONLY — never in the engine's abstractions, never
 * in src/backend.hh. The engine plane above speaks only backend.hh's
 * vendor-free surface; everything below speaks only the CUDA runtime:
 * one cudaStreamNonBlocking stream per lane, cudaEventRecord /
 * cudaStreamWaitEvent for ordering markers and gates, cudaLaunchHostFunc
 * for host callbacks, cudaMemcpyAsync / cudaMemcpyPeerAsync of exactly
 * the submitted span (the occupied span of the source chunk — the whole
 * class for a full chunk, the tail chunk's payload remainder), one
 * cudaEventDisableTiming event per provisioned handle — no timing on
 * the mint events, and no event or lane state read mid-flight anywhere;
 * the stamp mechanism keeps its own timing-enabled events per lane,
 * whose device-clock differences are read ONLY at the queue-close join,
 * post-barrier — the read delivered to the engine as facts, the one
 * blocking join remaining being the barrier at queue close and run end.
 * NOTHING here names logical chunks, tiles, plans, or scheduling —
 * lanes, bytes, and registrations only; every plan concept is the
 * engine plane's. Nothing is allocated after construction but the one
 * self-freeing gate box per wait_host_flag() enqueue and the stamp
 * provision's growth to its working set (recycled at the harvests, never
 * shrunk) — the same harness-fact discipline as the host lane's
 * unbounded FIFO.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include "backend.hh"
#include "host_worker.hh" // wait_host_flag's seam

#include <nnhaul/defs.h>

#include <cuda_runtime.h> // the vendor home of the whole backend

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nnhaul
{

// ---- registration : one body per (node, TaskId) ----

// The registered executable of one task symbol on one device node:
// the body enqueues the task's device work on the lane's OWN stream
// and returns — submission records and never blocks. Scalar
// arguments are registration-time facts of the task symbol,
// carried by the body's capture, never per-op plan fields. A body
// that fails to enqueue throws: a submission error the lane refuses
// at submission, never a silent skip.
using CudaLaunchFn = std::function<void(cudaStream_t, void *const *)>;

// ---- the per-device event provision ----

// One pool per device of events created at construction with
// cudaEventCreateWithFlags(cudaEventDisableTiming) — no timing is ever
// read. Minting never allocates: a minted handle IS one of these
// events, recorded on a lane. Handles recycle at barrier sweeps via
// release() only — the pool never auto-recycles (the engine's sweep
// decides retirement); destruction is likewise the engine's shutdown
// act, after the barriers.
class CudaEventPool final : public EventPool
{
  public:
    // `provision` events created on device `ordinal`'s context. A
    // provision of zero is a configuration fault, loud, before any
    // event exists (the host lane's same refusal). Construction is an
    // init-time act; after it nothing allocates.
    CudaEventPool(int ordinal, std::size_t provision);

    // Destroys the provision (only after the barriers — the engine's
    // shutdown ordering). A diagnostic stderr line per
    // failed destroy; a destructor never throws.
    ~CudaEventPool() override;

    CudaEventPool(CudaEventPool const &) = delete;
    CudaEventPool &operator=(CudaEventPool const &) = delete;
    CudaEventPool(CudaEventPool &&) = delete;
    CudaEventPool &operator=(CudaEventPool &&) = delete;

    // Extend the provision to new_provision events
    // (never shrink — loud on a smaller ask); the bitmap grows and
    // the new events are created on this device. Called by the engine
    // when the chunk pools grow (the ceiling's slot term grows with
    // them), engine thread only.
    void extend(std::size_t new_provision);

    // Leases one handle; exhaustion is a CONFIGURATION fault —
    // loud, with a census naming the provision and its sizing rule.
    // Lease discipline: acquire() and release() run on
    // the engine thread only (the owed derivation acquires, the
    // barrier sweep releases), so the free set carries no lock.
    EventHandle acquire() override;

    // Returns one handle to the provision — the barrier sweep's
    // recycle. Out-of-provision and double releases are bookkeeping
    // defects, loud, never serviced.
    void release(EventHandle e) override;

    // The batch-granular join (EventPool::join's CUDA half):
    // cudaEventSynchronize on the handle's event — the plan's OWN
    // tail, never the stream. An async failure delivers here as the
    // throw's text.
    void join(EventHandle e) override;

    // The CUDA event behind a handle, for this pool's lanes' mints
    // and waits. An out-of-provision handle is refused at submission.
    // No event state is read here or anywhere in this backend.
    cudaEvent_t raw(EventHandle e) const;

    std::size_t provision() const { return provision_; }
    int ordinal() const { return ordinal_; }

    // The sizing arithmetic, as the engine's init calls it: `m_p[i]` is
    // the op count of plan i (|COPY| + |TILE_OP| ops, at most one mint
    // per op); under depth-2 pipelining the typical outstanding set is
    // mints(p) + mints(p+1) <= 2 * max_p m_p, so the per-device
    // provision MUST be pow2ceil(2 * max_p m_p) FLOORED at 1024 (the
    // floor keeps small runs off the ceiling), NNHAUL_EVENT_POOL
    // overriding (a positive count; garbage is a loud init error). The
    // same rule sizes the host node's flag array. The displacement
    // ceiling — outstanding <= C_total * (1 + L), reachable only by
    // adversarial queues — is the number a deployment must plan for: a
    // deployment whose queues can reach it MUST raise NNHAUL_EVENT_POOL
    // to it at init. Pure: reads the knob, touches no device.
    static std::size_t suggest_event_provision(
        std::vector<std::size_t> const &m_p);

  private:
    int ordinal_;
    std::size_t provision_;
    std::vector<cudaEvent_t> events_;
    std::vector<std::uint64_t> used_; // the lease bitmap, provision bits
    std::size_t used_count_ = 0;
    std::size_t hint_ = 0; // the word the first-fit scan restarts at
    // The storage race's guard: the host-mint record twin reads
    // raw(e) on the HOST WORKER thread (its FIFO position), while
    // extend() — the engine thread's growth commit — resizes
    // events_. mint/wait/join also call raw(), but on the engine
    // thread (extend's own thread, uncontended); the mutex exists
    // for the worker's cross-thread read.
    mutable std::mutex events_mu_;
};

// ---- the shared stream tail of the three lane realizations ----

// The CUDA realization of ordering over ONE non-blocking stream of
// one device: mint = cudaEventRecord on the
// stream, wait = cudaStreamWaitEvent, notify = cudaLaunchHostFunc,
// barrier = cudaStreamSynchronize — the queue-close/run-end join
// only, and the detection point for a faulted op's async error:
// delivered from barrier() with a census (the ENGINE wraps it with
// the op's own census, per the interface seam). Not a lane itself and
// not part of backend.hh's surface — the composition root of the
// three public realizations below; it names no plan concept.
class CudaLaneOps
{
  public:
    // Makes device `ordinal`'s primary context current (idempotent:
    // the engine's init made contexts current first)
    // and creates the lane's ONE stream with
    // cudaStreamCreateWithFlags(..., cudaStreamNonBlocking) in that
    // context. `events` is the device's CudaEventPool
    // — an unowning reference the engine must outlive every use: it
    // constructs the pool before the lanes and destroys it after the
    // lanes' barriers.
    CudaLaneOps(int ordinal, CudaEventPool &events);
    ~CudaLaneOps(); // stream destroy; a diagnostic, never a throw

    CudaLaneOps(CudaLaneOps const &) = delete;
    CudaLaneOps &operator=(CudaLaneOps const &) = delete;
    CudaLaneOps(CudaLaneOps &&) = delete;
    CudaLaneOps &operator=(CudaLaneOps &&) = delete;

    // cudaEventRecord(events.raw(e), stream) — an ordering marker
    // completing when everything enqueued on the lane before it
    // completes. A failure is refused at submission, loudly.
    void mint(EventHandle e);

    // cudaStreamWaitEvent(stream, e, 0) — a gate holding the
    // lane's later work until the mint has completed. CUDA's wait
    // captures the event's state AT CALL TIME and the engine
    // guarantees every waited mint was enqueued strictly earlier.
    void wait(EventHandle e);

    // The pool-aware wait: the handle names a slot in `pool`, which may
    // belong to another device (a peer COPY waits the source's mint).
    void wait_from(EventPool &pool, EventHandle e);

    // cudaLaunchHostFunc(stream, fn, arg) — the once-executed host
    // callback at this lane position, after earlier-enqueued work and
    // ordering later-enqueued work. A null hook is refused at
    // submission. The hook must not call CUDA APIs and must not block
    // beyond its own bounded work (the host-flag gates ride this
    // mechanism — a backend-internal mechanism; the tracer's stamps ride
    // the issue-position clock reads, no hook exists for them).
    void notify(NotifyHook fn, void *arg);

    // ---- the stamp mechanism, the device-lane realization: the HOST
    //      clock read at the op's ISSUE position. The rationale: a
    //      stream-ordered host function stalls its lane for the duration
    //      of host dispatch, and a TIMING EVENT record — though it
    //      stalls no work — taxes the HOST submission path, whose wall
    //      cost per record grows with the stream's pending depth, while
    //      the device-side cost stays small; either per-op insertion
    //      taxes a path the timing budget cannot spare. The issue read
    //      taxes nothing: the submission into the lane IS in the lane's
    //      order by construction, so one host-clock read per op at its
    //      submission places the fact AT the lane position, and the
    //      op's measured interval is the TILE it owns of the lane's
    //      issue order — [its stamp, the next stamp on that lane] — the
    //      duration including the lane's queue wait (the NNHaul
    //      precedent). The provision is the lane's own: clock-read slots
    //      grown to the working set and recycled at the harvests. The
    //      pair read is the queue-close join's act: the two slots'
    //      values; an unwritten slot is the barrier's own defect, loud
    //      (the join is the detection point).

    // Lease one stamp slot. Engine thread. Growth is the
    // harness-fact discipline; a failed growth is the loud
    // configuration fault, never serviced.
    EventHandle stamp_acquire();

    // Return one stamp slot — the harvest's act, after the tile was
    // composed. Out-of-provision and double releases are bookkeeping
    // defects, loud, never serviced.
    void stamp_release(EventHandle h);

    // The clock read AT this lane position: the submission's instant,
    // on the thread that submits (the engine's issue order — the
    // engine thread direct, or the bridge's waiter in the same
    // order). No CUDA API here — the read taxes nothing.
    void stamp(EventHandle h);

    // The pair's fact at the queue-close join (post-barrier): the two
    // slots' steady-clock reads — the tile this op owns of the lane's
    // issue order. A slot of zero is unwritten — the barrier's own
    // defect, loud.
    StampFact stamp_read(EventHandle before, EventHandle after);

    // The lane's whole tail joined — queue close and run end
    // only; a faulted op's async failure is delivered HERE, the
    // detection point, with a census naming the device lane. The
    // per-lane deferral bridge joins
    // through the EXECUTOR's light close — per-lane drain +
    // terminal-event join — never this whole-stream sync, which
    // stands for teardown and forced joins; dependents never run
    // past the fault inside this stream's order.
    void barrier();

    // The DEVICE-lane-waits-host-mint mechanism, an implemented but
    // unused alternative: a cudaLaunchHostFunc on this lane's stream
    // blocking on the host worker's mint flag. NOT the default —
    // empirically it wedges the driver under pipelined submission on
    // this stack (the DeferralBridge's comment carries the finding) —
    // kept as an option. HAZARD: if the host lane faults so the mint
    // never fires, this lane's barrier waits forever.
    void wait_host_flag(HostWorkerLane &host, EventHandle host_e);


    cudaStream_t stream() const { return stream_; }
    int ordinal() const { return ordinal_; }

  private:
    int ordinal_;
    CudaEventPool *events_; // unowning; the engine's member order rules
    cudaStream_t stream_ = nullptr;

    // The stamp provision (the stamp mechanism, the device-lane
    // realization): the issue positions' clock reads, steady ns, 0 =
    // not written (yet). Empty until the first stamp — an untraced
    // run allocates nothing here. The reads come from the submitting
    // thread (the engine thread direct, or the bridge's waiter in
    // the same order); the pair read runs on the engine thread at
    // the close — the barriers order every written slot before it.
    std::vector<std::uint64_t> stamp_values_;
    std::vector<bool> stamp_leased_; // the lease bitmap, provision bits
    std::vector<EventHandle> stamp_free_;
    // The provision is read/written by the engine thread (acquire/release,
    // pair reads) and by the bridge waiter thread (stamp records). Growth
    // must not reallocate under a waiter's in-flight record.
    mutable std::mutex stamp_mu_;
};

// ---- the device node's execution lane and arena ----

// One cudaStreamNonBlocking stream per device node, created
// at construction in that device's context, owning the device's
// arena (grown by allocation segments) — whose chunk addresses are
// valid and stable for the run's life and DMA-stable. launch()
// resolves TaskId through the per-node registration table
// (register_task, an init-time act); an unregistered task is refused
// loudly at launch — an enqueue detection point (the checker's
// task-not-runnable closes the class at check time; this refusal is
// the residual backstop).
class CudaExecLane final : public ExecLane
{
  public:
    // The context binding,
    // the stream, and a SEGMENTED arena grown by the two-phase
    // ladder. `limit_chunks` is the SOFT ceiling (0 = unlimited —
    // growth ends only at the non-allocable flip); `initial` is the
    // explicit preallocation head start (default 0), CHECKED against
    // the limit (a larger ask is a loud init fault) and honored
    // loudly at init (an ask the device cannot fully back is a
    // configuration fault). Ordinary growth later stops at the soft
    // wall or the flip, never thrown.
    CudaExecLane(int ordinal, std::size_t limit_chunks,
        std::size_t initial, CudaEventPool &events);

    // The TWO-PHASE allocation rule: in the
    // ALLOCABLE phase every event tries TILE-WISE first and falls
    // back to SINGLE chunks for that event only; a failed
    // single-chunk allocation flips the pool NON-ALLOCABLE,
    // permanently. The soft limit bounds every event; nothing is
    // freed until the lane's death.
    std::size_t alloc_event(std::size_t chunks);

    // The planning-time growth: the tile through the ladder, then the
    // remaining budget as one more event under a soft limit. Returns
    // how many chunks were added.
    std::size_t grow_to_wall(std::size_t tile_chunks);

    // ENSURE wrapper for the init-time preallocation ask.
    std::size_t grow(std::size_t n);

    // The contiguous span from chunk `chunk` to its
    // SEGMENT's end, in bytes — merged copies must stay inside one
    // allocation.
    std::size_t contig_bytes(ChunkId chunk) const;

    bool hard_capped() const { return hard_capped_; }
    std::size_t limit_chunks() const { return limit_; }

    // Releases the arena and the stream (the engine's shutdown
    // barriers precede destruction). Diagnostics on failure;
    // never throws.
    ~CudaExecLane() override;

    CudaExecLane(CudaExecLane const &) = delete;
    CudaExecLane &operator=(CudaExecLane const &) = delete;
    CudaExecLane(CudaExecLane &&) = delete;
    CudaExecLane &operator=(CudaExecLane &&) = delete;

    // The chunk table entry — VALID and STABLE from the chunk's
    // allocation to the run's death, directly sourceable
    // and sinkable by the transfer engines (native device bytes; a
    // transfer that bounce-buffers them is non-conformant). An
    // out-of-pool chunk is a defect, loud. The NodeId is this lane's
    // own device — the arena is per node, so the address needs only
    // the chunk.
    void *addr(ChunkId chunk) const;

    std::size_t cardinality() const { return chunks_.size(); }
    std::size_t segment_count() const // driver allocations (one per grow)
    {
        return segments_.size();
    }
    int ordinal() const { return ops_.ordinal(); }
    // The lane's own stream — the binding target (the
    // per-device library handles the engine binds to this lane's
    // stream at init).
    cudaStream_t stream() const { return ops_.stream(); }

    // The registration surface: one body per TaskId on this node.
    // A null body or a double registration is a loud census — one
    // symbol, one body, resolved at INIT.
    void register_task(TaskId task, CudaLaunchFn fn);

    // Appends the registered body's enqueues on the lane's stream.
    // An unregistered TaskId is refused loudly at submission; a
    // body's own failed enqueue propagates as the same submission
    // refusal at submission. `operand_addrs` is the engine-owned
    // array, handed to the body verbatim.
    void launch(
        TaskId task, NodeId node, void *const *operand_addrs) override;

    void mint(EventHandle e) override { ops_.mint(e); }
    void wait(EventHandle e) override { ops_.wait(e); }
    void wait_from(EventPool *pool, EventHandle e) override
    {
        if (pool == nullptr)
        {
            ops_.wait(e);
            return;
        }
        ops_.wait_from(*pool, e);
    }
    void notify(NotifyHook fn, void *arg) override
    {
        ops_.notify(fn, arg);
    }
    EventHandle stamp_acquire() override { return ops_.stamp_acquire(); }
    void stamp_release(EventHandle h) override
    {
        ops_.stamp_release(h);
    }
    void stamp(EventHandle h) override { ops_.stamp(h); }
    StampFact stamp_read(EventHandle before, EventHandle after) override
    {
        return ops_.stamp_read(before, after);
    }
    void barrier() override { ops_.barrier(); }

    // The device-waits-host-mint gate (CudaLaneOps's
    // mechanism, this lane's stream).
    void wait_host_flag(HostWorkerLane &host, EventHandle host_e)
    {
        ops_.wait_host_flag(host, host_e);
    }


    // The SOFT-CEILING precedence for this device's arena growth, in
    // BYTES (see the .cc's note): NNHAUL_LIMIT_CUDA_MEM (MiB) wins
    // when set; else a positive init argument; else 0 =
    // unlimited-until-hard-failure. A SUB-CLASS cap is a loud init
    // error: every tile costs a whole chunk. Pure — it reads only the
    // env knob and the argument, and touches no device.
    static std::size_t suggest_limit_bytes(
        int ordinal, std::size_t api_cap_bytes,
        std::size_t fixed_reserve_bytes = 0);

  private:
    CudaLaneOps ops_;
    // The segmented arena: segments appended on demand by
    // the two-phase ladder (a tile's chunks in one allocation, single
    // chunks in the per-event fallback). A chunk's address is VALID
    // and STABLE from its allocation to the run's death (a chunk never
    // moves); chunk ids index the table; a segment's chunks are
    // VA-contiguous.
    struct Segment
    {
        void *base = nullptr;
        std::size_t chunks = 0;
    };
    std::vector<Segment> segments_;
    std::vector<void *> chunks_;
    std::size_t limit_ = 0; // the soft ceiling (0 = unlimited)
    bool hard_capped_ = false;

    // One allocation attempt of exactly `seg_chunks` chunks; false
    // WITHOUT the wall decision (the caller owns it).
    bool allocate_segment(std::size_t seg_chunks);

    // The mode flip: a single-chunk allocation failed — non-allocable
    // permanently. One diagnostic line per process.
    void enter_non_allocable();
    std::unordered_map<TaskId, CudaLaunchFn> tasks_;

  public:
};

// ---- the device-host transfer channel ----

// The direction of a host-endpoint channel: HostToDevice feeds the
// device node (h2d), DeviceToHost drains it (d2h) — one channel per
// ordered pair, both directions always present.
enum class HostChannelDirection : std::uint8_t
{
    HostToDevice,
    DeviceToHost
};

// One stream per host-endpoint ordered pair, the stream belonging
// to the pair's single device; one copy IS one cudaMemcpyAsync of
// EXACTLY the submitted span over the host node's pinned bytes (the
// host pool's DMA-stable memory) — the occupied span of
// the source chunk, 1..chunk_class_bytes() bytes.
class CudaHostChannel final : public TransferChannel
{
  public:
    CudaHostChannel(
        int ordinal, HostChannelDirection dir, CudaEventPool &events);
    ~CudaHostChannel() override;

    CudaHostChannel(CudaHostChannel const &) = delete;
    CudaHostChannel &operator=(CudaHostChannel const &) = delete;
    CudaHostChannel(CudaHostChannel &&) = delete;
    CudaHostChannel &operator=(CudaHostChannel &&) = delete;

    // One cudaMemcpyAsync of exactly `nbytes` — the occupied span the
    // engine submitted — kind by the pair's direction. A span outside
    // 1..chunk_class_bytes(), or a submission failure, is loud at
    // submission; never silently clipped.
    void copy(void const *src, void *dst, std::size_t nbytes) override;

    // ONE cudaMemcpyAsync for a merged run of contiguous
    // occupied spans (the executor's COPY-run realization): the byte
    // count is the run's SUM — over one class, never a per-chunk count
    // — and the same kind by the pair's direction. Zero bytes, or a
    // submission failure, is loud at submission.
    void copy_run(
        void const *src, void *dst, std::size_t nbytes) override;

    void mint(EventHandle e) override { ops_.mint(e); }
    void wait(EventHandle e) override { ops_.wait(e); }
    void wait_from(EventPool *pool, EventHandle e) override
    {
        if (pool == nullptr)
        {
            ops_.wait(e);
            return;
        }
        ops_.wait_from(*pool, e);
    }
    void notify(NotifyHook fn, void *arg) override
    {
        ops_.notify(fn, arg);
    }
    EventHandle stamp_acquire() override { return ops_.stamp_acquire(); }
    void stamp_release(EventHandle h) override
    {
        ops_.stamp_release(h);
    }
    void stamp(EventHandle h) override { ops_.stamp(h); }
    StampFact stamp_read(EventHandle before, EventHandle after) override
    {
        return ops_.stamp_read(before, after);
    }
    void barrier() override { ops_.barrier(); }


    int ordinal() const { return ops_.ordinal(); }
    HostChannelDirection direction() const { return dir_; }

  private:
    CudaLaneOps ops_;
    HostChannelDirection dir_;
};

// ---- the device-device transfer channel ----

// One stream per probed device pair, the stream belonging to the
// pair's DESTINATION device (cudaMemcpyPeerAsync accepts it); one copy
// IS one cudaMemcpyPeerAsync of exactly the submitted span — the
// occupied span of the source chunk, 1..chunk_class_bytes() bytes. The
// pair was probed and peer-enabled at init (probe_peer_matrix /
// enable_peer_access); this class never introspects the devices itself
// — the probe is the run's ONLY device introspection.
class CudaPeerChannel final : public TransferChannel
{
  public:
    // The stream is created on `dst_ordinal`'s device. A self-pair is
    // refused loudly — a channel needs two distinct endpoints.
    CudaPeerChannel(
        int src_ordinal, int dst_ordinal, CudaEventPool &events);
    ~CudaPeerChannel() override;

    CudaPeerChannel(CudaPeerChannel const &) = delete;
    CudaPeerChannel &operator=(CudaPeerChannel const &) = delete;
    CudaPeerChannel(CudaPeerChannel &&) = delete;
    CudaPeerChannel &operator=(CudaPeerChannel &&) = delete;

    // One cudaMemcpyPeerAsync of exactly `nbytes` — the occupied span
    // the engine submitted — from the source device's chunk address
    // to the destination device's, on the destination device's stream.
    // A span outside 1..chunk_class_bytes(), or a submission failure, is
    // loud at submission; never silently clipped.
    void copy(void const *src, void *dst, std::size_t nbytes) override;

    // ONE cudaMemcpyPeerAsync for a merged run of
    // contiguous occupied spans: the run's byte SUM, over the probed
    // pair, on the destination device's stream. Zero bytes, or a
    // submission failure, is loud at submission.
    void copy_run(
        void const *src, void *dst, std::size_t nbytes) override;

    void mint(EventHandle e) override { ops_.mint(e); }
    void wait(EventHandle e) override { ops_.wait(e); }
    void wait_from(EventPool *pool, EventHandle e) override
    {
        if (pool == nullptr)
        {
            ops_.wait(e);
            return;
        }
        ops_.wait_from(*pool, e);
    }
    void notify(NotifyHook fn, void *arg) override
    {
        ops_.notify(fn, arg);
    }
    EventHandle stamp_acquire() override { return ops_.stamp_acquire(); }
    void stamp_release(EventHandle h) override
    {
        ops_.stamp_release(h);
    }
    void stamp(EventHandle h) override { ops_.stamp(h); }
    StampFact stamp_read(EventHandle before, EventHandle after) override
    {
        return ops_.stamp_read(before, after);
    }
    void barrier() override { ops_.barrier(); }


    int src_ordinal() const { return src_ordinal_; }
    int dst_ordinal() const { return ops_.ordinal(); }

  private:
    CudaLaneOps ops_; // the stream lives on the destination device
    int src_ordinal_;
};

// ---- the init-time helpers  ----

// Device selection: parse NNHAUL_DEVICES — comma-separated CUDA
// ordinals — into the ordinals behind NodeIds 0..k-1 in ASCENDING
// order (device node i IS the i-th smallest selected ordinal, whatever
// order the knob listed them in). The default is
// ordinals 0..ncuda-1 ascending. A listing whose count differs from
// `ncuda` is a LOUD init error — the knob selects WHICH k devices,
// `ncuda` says HOW MANY; they MUST agree. Garbage
// tokens, an empty listing, an impossibly large ordinal, and a
// duplicate ordinal are loud errors too. PURE: no device is touched —
// ordinal-range validation against the driver belongs to the probe,
// the run's only device introspection.
std::vector<int> select_devices(int ncuda);

// The peer probe: cudaDeviceCanAccessPeer
// per ordered pair over `ordinals`, returned as k*k row-major bits —
// bit[i*k + j] set iff SOURCE i can access DESTINATION j. The
// diagonal is false, an unidirectional answer is recorded FAITHFULLY,
// never mirrored, and an absent pair is simply absent (no hardware
// path, no channel — staging is planned dataflow, the engine's
// business). Ordinals outside the driver's device count are a loud
// census. THIS — plus nothing else after init — is the run's ONLY
// device introspection.
std::vector<std::uint8_t> probe_peer_matrix(
    std::vector<int> const &ordinals);

// Peer enable: cudaDeviceEnablePeerAccess on every
// probed pair, AFTER the contexts — the caller made each device's
// primary context current first, and this helper re-binds
// the ACCESSING (source) device current before each call because the
// call requires it — exactly why enable cannot sit inside the probe
// step: no context exists yet at probe time. Already-enabled pairs are
// success-shaped and skipped. A failed enable is a loud census
// naming the ordered pair; a `peer_bits` of the wrong shape is a
// caller defect, loud.
void enable_peer_access(std::vector<int> const &ordinals,
    std::vector<std::uint8_t> const &peer_bits);

} // namespace nnhaul
