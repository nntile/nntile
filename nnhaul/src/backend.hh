/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * abstract backend surface: a fixed set of backend operations rendered
 * as vendor-free C++ interfaces — the whole engine-side surface of the
 * backend contract.
 *
 * @file src/backend.hh
 * The abstract backend surface: EventHandle and NotifyHook, StampFact,
 * ExecLane, TransferChannel, and EventPool. The surface is CLOSED: the
 * engine uses exactly these operations and nothing more, speaks only
 * this surface, and every realization speaks only its own runtime — no
 * vendor header is included anywhere below. Two disciplines bind every
 * lane and the engine plane alike, and this header states them once:
 * the ENGINE plane never queries, polls, or waits on an event or on
 * lane progress, and host knowledge comes only from notifications, the
 * stamp facts read at the queue-close join, and barriers; and every
 * enqueue records the request and returns, never blocking on and never
 * observing the state of any earlier work on any lane. The only
 * engine-initiated blocking act is the barrier. EventHandle has exactly
 * ONE home, this file: sibling types that need it (the owed registry of
 * src/dependency.hh among them) include this definition and never
 * restate it.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include "plan.hh" // TaskId; NodeId via inventory.hh

#include <cstdint>
#include <stdexcept>
#include <string>

namespace nnhaul
{

// ---- the abstract backend surface ----

// An event handle IS an index into a provisioned pool — opaque
// to the engine, provisioned at init and recycled at barrier sweeps.
// A handle is minted and waited as LANE appends — an ordering marker
// completing when everything enqueued before it completes; it carries
// no payload, no timestamp, and NO readable state, and the engine
// plane never queries, polls, or waits on one.
using EventHandle = std::uint32_t;

// NotifyHook IS the once-executed host callback at a lane position
// — after earlier-enqueued work, ordering later-enqueued work. The
// host-flag gates ride this mechanism; a hook must not throw (noexcept)
// and must not block its lane beyond its own bounded work.
using NotifyHook = void (*)(void *arg) noexcept;

// The stamp fact: the pair read's product — the op's start and end
// clock reads in the ONE steady-clock domain the trace renders, ns.
// Every lane's read is the HOST clock taken at the op's issue position
// (stated at each realization); the host lane's worker reads its own
// clock at its FIFO positions.
struct StampFact
{
    std::uint64_t start_ns = 0;
    std::uint64_t end_ns = 0;
};

// One execution lane per node, with FULL serialization — the lane's op
// N+1 does not BEGIN until op N has COMPLETED, begin-after-completion
// with memory-ordering strength, not merely completion in enqueue
// order.
class EventPool;

class ExecLane
{
  public:
    virtual ~ExecLane() = default;

    // Appends a TILE_OP's execution. Task identity -> executable is
    // resolved at INIT (registration); scalar arguments are
    // registration-time facts of the task symbol, never per-op plan
    // fields; `operand_addrs` is the engine-resolved array of operand
    // chunk addresses. Submission records and returns.
    virtual void launch(
        TaskId task, NodeId node, void *const *operand_addrs) = 0;

    // Appends an ordering marker immediately AFTER the op's device
    // work, completing when everything enqueued on the lane before it
    // completes. Minting never allocates; handles come from the
    // init-time provision.
    virtual void mint(EventHandle e) = 0;

    // Appends a gate immediately BEFORE the consumer's own device
    // work, holding the lane's later work until the event's mint has
    // completed. Cross-lane only — same-lane waits are elided and are
    // never issued; the engine guarantees every waited mint was
    // enqueued strictly earlier.
    virtual void wait(EventHandle e) = 0;

    // Wait for a handle minted in a specific pool. A handle is an index
    // into its HOME pool, which need not be this lane's pool: a peer COPY
    // lane waits events minted by the source device's pool. The default
    // keeps the historical lane-local resolution for lanes whose waits
    // are always in their own family; CUDA lanes override it.
    virtual void wait_from(EventPool * /*pool*/, EventHandle e)
    {
        wait(e);
    }

    // Appends the once-executed host callback at this lane position
    // — after earlier-enqueued work, ordering later-enqueued work. The
    // host-flag gates ride this mechanism.
    virtual void notify(NotifyHook fn, void *arg) = 0;

    // ---- the stamp mechanism: the per-op duration facts. A stamp IS
    //      a clock read at a lane position: leased from the lane's own
    //      provision, recorded AT its lane position, and read as a
    //      pair ONLY at the queue-close join, after the barriers that
    //      retired the position (the pair's completion IS the op's
    //      completion fact). Stamps are facts, never inputs: they
    //      never gate, skip, or reorder an op. EVERY lane realizes the
    //      read as the HOST clock read at the op's ISSUE position — the
    //      submission into the lane IS in the lane's order — and the
    //      op's measured interval is the TILE it owns of the lane's
    //      issue order, [its stamp, the next stamp on that lane] (the
    //      plan's tail stamp closes the last tile; the duration
    //      includes the lane's queue wait, the established precedent — the
    //      cost rationale for not inserting per-op CUDA work is in
    //      src/plan_trace.hh). The host lane's worker reads its own
    //      clock at the FIFO position.

    // Lease one stamp from the lane's provision. The provision grows
    // to the working set (the un-retired plans' stamps) and recycles
    // at the harvests; a lease that cannot grow is a loud
    // configuration fault, never serviced.
    virtual EventHandle stamp_acquire() = 0;

    // Return one stamp to the provision — the harvest's act, after
    // the pair was read. An out-of-provision or doubly released stamp
    // is a bookkeeping defect, loud, never serviced.
    virtual void stamp_release(EventHandle h) = 0;

    // Record the stamp AT this lane position: the clock read that
    // falls immediately before or after the op's device work.
    // Submission records and returns; recording never gates, never
    // observes, never blocks on earlier work.
    virtual void stamp(EventHandle h) = 0;

    // The pair's fact, read ONLY at the queue-close join (after the
    // barriers that retired both positions): start and end in the one
    // steady-clock domain. An incomplete pair HERE is the barrier's
    // own defect — loud, never serviced (the join is the detection
    // point; a stamp the barriers failed to retire is a barrier
    // defect).
    virtual StampFact stamp_read(EventHandle before, EventHandle after) = 0;

    // Joins the lane's whole tail. The engine raises barriers only at
    // queue close and run end.
    virtual void barrier() = 0;
};

// One transfer channel per connected ORDERED node pair, with the same
// full serialization as an execution lane. The four ordering methods
// are ExecLane's surface VERBATIM — realized here by public
// inheritance, the literal form: a channel IS a lane for every
// ordering operation, and the engine's lane-walking code may hold both
// kinds as ExecLane. Exactly one ExecLane method does not belong to a
// channel — launch (a TILE_OP maps to an execution lane only) — and it
// is closed below with a single final refusal rather than left
// abstract for every realization to re-refuse.
class TransferChannel : public ExecLane
{
  public:
    virtual ~TransferChannel() = default;

    // Appends one copy request moving exactly `nbytes` — the OCCUPIED
    // SPAN of the source physical chunk (tile_chunk_span): the whole
    // chunk class for a full chunk, the tile payload's tail remainder
    // for a tail chunk — and nothing more, never the wasted bytes of
    // the block behind the tail. The count is the engine's submission
    // fact, computed from the linked logical chunk's registered tile
    // size; the channel moves exactly what it is handed. Both addresses
    // are backend-owned pool bytes, DMA-stable. A span of zero, or one
    // beyond the chunk class, is a construction/submission fault —
    // loud below, never silently clipped.
    virtual void copy(
        void const *src, void *dst, std::size_t nbytes) = 0;

    // The RUN realization of consecutive same-lane COPY ops: ONE
    // request moving the SUM of occupied spans whose addresses are
    // contiguous on BOTH endpoints (the executor proves the contiguity
    // address-exactly before merging, so the run moves the same bytes
    // over the same addresses in the same lane order — bitwise-equal by
    // construction; the plan's per-chunk COPY ops, their issue log,
    // their trace events, and their span accounting all stay untouched
    // — this is the BACKEND's realization of the queue, never a
    // plan-content change). A span of zero is a submission fault, loud.
    // The DEFAULT realizes a run as class-sized slices through copy() —
    // an interior span of a merged run is always whole-class (a tail
    // span is never followed by a contiguous address), so the slicing
    // is exact for every realization; the CUDA channels override it
    // with the one driver call (fewer cudaMemcpyAsync entries
    // contending on the driver's global locks).
    virtual void copy_run(
        void const *src, void *dst, std::size_t nbytes)
    {
        while (nbytes > 0)
        {
            std::size_t const n = nbytes < chunk_class_bytes()
                ? nbytes
                : chunk_class_bytes();
            copy(src, dst, n);
            src = static_cast<char const *>(src) + n;
            dst = static_cast<char *>(dst) + n;
            nbytes -= n;
        }
    }

    // A TILE_OP never lands on a transfer lane; a launch
    // here is an engine contract fault, refused loudly at submission
    // — never serviced.
    void launch(TaskId, NodeId, void *const *) override final
    {
        throw std::runtime_error(
            "transfer channel: launch of a TILE_OP refused — TILE_OP "
            "maps to an execution lane only");
    }

  protected:
    // The copy span's belt, shared by every realization: the occupied
    // span of a physical chunk is tile_chunk_span's range — 1 to
    // chunk_class_bytes() bytes (a tail chunk's remainder fills the
    // block's head, never its whole absence). A count outside that
    // range is an engine defect at submission: loud, never clipped to
    // fit, never serviced.
    static void require_copyable_span(std::size_t nbytes)
    {
        if (nbytes == 0 || nbytes > chunk_class_bytes())
        {
            throw std::runtime_error(
                "transfer channel: copy span of " +
                std::to_string(nbytes) + " bytes refused — a COPY moves "
                "the occupied span of one physical chunk, 1.." +
                std::to_string(chunk_class_bytes()) +
                " bytes ; a zero or over-class span "
                "is a construction/submission error, never clipped");
        }
    }
};

// The init-time event provision, recycled at barrier sweeps.
// Exhaustion is a CONFIGURATION fault, caught loudly with a census.
class EventPool
{
  public:
    virtual ~EventPool() = default;

    // Leases one handle; a pool that cannot is a configuration
    // fault — loud, with a census naming the provision (never
    // serviced, never retried).
    virtual EventHandle acquire() = 0;

    // Returns one handle to the provision (a barrier sweep); a
    // bookkeeping defect — an out-of-provision or a doubly released
    // handle — is loud, never serviced.
    virtual void release(EventHandle e) = 0;

    // Joins ONE handle's mint — blocks the calling (engine) thread
    // until the event recorded for this handle completed on its
    // lane's stream, throwing the lane's fault text when the work
    // under it failed. THE BATCH-GRANULAR BARRIER: the close joins the
    // PLAN's OWN terminal event, never the lane's whole stream — the
    // successor plan's already-fed prefix sits behind the event in
    // stream order and cannot delay the join. An async failure's text
    // names the device; the executor wraps it with the census.
    virtual void join(EventHandle e) = 0;
};

} // namespace nnhaul
