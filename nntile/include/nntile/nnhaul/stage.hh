/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/nnhaul/stage.hh
 * Universal staging fallback for classic NNTile ops on chunked NNHaul
 * tiles (docs/chunked-tiles.md §4.3): gather chunked buffers into
 * contiguous scratch handles, run the original single-pointer codelet,
 * scatter the results back.
 *
 * @version 1.1.0
 * */

#pragma once

#ifdef NNTILE_USE_NNHAUL

#include <nntile/base_types.hh>
#include <nntile/nnhaul/chunk_gate.hh>
#include <nntile/nnhaul/codelet.hh>
#include <nntile/nnhaul/ops/copy.hh>

#include <nnhaul/nnhaul.hh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nntile
{
namespace haul
{
namespace stage
{

// ---- chunked-handle detection ------------------------------------------
//
// NNHaul's public Handle API exposes no chunk query at submit time (the
// chunk table only materializes on the worker's Buffer snapshot inside a
// kernel), so detection reconstructs the allocation decision from two
// sources:
//
// 1. The registry (exact): every chunked handle whose producer is known
//    to NNTile registers itself here - core::Tile::alloc_handle for
//    graph tiles, note_chunked() for raw two-argument handles built by
//    embedders and tests outside the Tile allocation path.
// 2. The policy recompute (conservative): an unknown handle is treated
//    as chunked when the classic gate is open and the nnhaul ladder
//    would chunk its nbytes. This is exactly Tile::alloc_handle's
//    decision, so it is exact for tile payloads; for foreign handles it
//    can only err towards staging, which is correct (staging is correct
//    for unchunked buffers too, just slower). Runtime::compile() throws
//    before flipping the gate off while chunked tile payloads live, so
//    the recompute can never miss a still-allocated chunked tile.

//! Remember a chunked handle for handle_is_chunked() and the compile-time
//! torch gate. Idempotent. Call before submitting tasks that use it.
//! chunk_bytes (when the producer knows it - Tile::alloc_handle and the
//! test factories do) is recorded for handle_chunk_bytes(): submit-time
//! launch-count guards need the handle's REAL chunk class, which the
//! policy recompute can only approximate for foreign handles.
void note_chunked(::nnhaul::Handle &handle,
    std::size_t chunk_bytes = 0);

//! Drop a remembered handle (idempotent). Must run before the handle is
//! destroyed so a recycled address cannot alias a stale entry.
void forget_handle(::nnhaul::Handle &handle);

//! True when a contiguous-pointer codelet must stage this handle.
bool handle_is_chunked(::nnhaul::Handle &handle);

//! The handle's chunk size in bytes: the recorded value for announced
//! handles, the policy recompute (Tile::alloc_handle's exact decision)
//! for unknown ones, 0 when the handle is not chunked. Submit-time
//! geometry guards (transpose's degenerate-panel guard) read this.
std::size_t handle_chunk_bytes(::nnhaul::Handle &handle);

//! Number of live remembered chunked handles (the torch-op gate reads
//! this at compile time).
std::size_t chunked_handle_count();

//! How many submits took the staged path so far (the engagement probe
//! for the chunked-tile tests).
inline std::atomic<std::uint64_t> g_staged_submit_count{0};

//! Engagement probe counter (see staged_submit_count()).
inline void note_staged_submit() noexcept
{
    g_staged_submit_count.fetch_add(1, std::memory_order_relaxed);
}

//! How many submits took the staged path so far.
inline std::uint64_t staged_submit_count() noexcept
{
    return g_staged_submit_count.load(std::memory_order_relaxed);
}

//! How many scratch->orig scatters were submitted so far (every staged
//! buffer whose merged access writes - the W/RW engagement probe for
//! the pure-Read scatter skip).
inline std::atomic<std::uint64_t> g_staged_scatter_count{0};

//! Engagement probe counter (see staged_scatter_count()).
inline void note_staged_scatter() noexcept
{
    g_staged_scatter_count.fetch_add(1, std::memory_order_relaxed);
}

//! How many scratch->orig scatters were submitted so far.
inline std::uint64_t staged_scatter_count() noexcept
{
    return g_staged_scatter_count.load(std::memory_order_relaxed);
}

//! How many pure-Read scatters were SKIPPED so far: staged buffers
//! whose merged access is Read-only never copy their scratch back
//! (the gather left the orig bytes untouched), so the op's reads stay
//! non-destructive - the orig's version does not advance and its next
//! eviction stays transferless. The skip probe for the chunked-tile
//! tests.
inline std::atomic<std::uint64_t> g_staged_read_skip_count{0};

//! Engagement probe counter (see staged_read_skip_count()).
inline void note_staged_read_skip() noexcept
{
    g_staged_read_skip_count.fetch_add(1, std::memory_order_relaxed);
}

//! How many pure-Read scatters were skipped so far.
inline std::uint64_t staged_read_skip_count() noexcept
{
    return g_staged_read_skip_count.load(std::memory_order_relaxed);
}

// ---- scratch generations ------------------------------------------------
//
// Scratch lifetime scheme (the provably-safe part):
//
// Every staged submit queues its scratch handles into a live set. The
// live set is destroyed ONLY when the task queue is known idle -
// immediately after nnhaul::wait() returned (starpu_task_wait_for_all_
// counted drains it; see sync_defer.hh) - or, as the process-exit
// fallback, after nnhaul::shutdown() when no runtime exists anymore.
//
// Rationale: ~Handle (nnhaul src/api.cc ~428-455) places an invalidate
// marker EXACTLY AFTER the handle's last queued task and waits for the
// sweep (dead_done), which by itself makes destruction use-after-free
// safe. But the sweep's quietness test (rt.hh sweepable()) counts
// in-flight kernels through the handle's CURRENT replica generation;
// a replica retired and rebuilt while kernels of an earlier generation
// are still in flight can look quiet before those kernels complete.
// Destroying a handle while the single in-order stream is BUSY can
// therefore race the sweep against the last user's completion inside
// nnhaul's worker (an nnhaul-internal hazard this header must not
// poke). Destroying only at wait()-idle closes the window entirely:
// every task already completed, so the marker sweeps immediately and
// the destructor's wait is a no-op poll. Between waits, the queued
// scratch handles are ordinary offloadable handles - cap pressure
// evicts their idle replicas fractionally, exactly like tile payloads,
// so keeping them alive costs no pinned device memory.
// nnhaul::shutdown() releases the blocks of still-registered handles
// itself, so the exit fallback (destruction after shutdown) is also
// safe.

//! Queue the scratch handles of the just-submitted staged op into the
//! live set (moved in; destroyed by the next drain).
void stash_scratches(
    std::vector<std::unique_ptr<::nnhaul::Handle>> scratches);

//! Destroy every queued scratch. Call only when the task queue is known
//! idle, i.e. immediately after nnhaul::wait() returned.
void drain_scratches();

// ---- staged submission --------------------------------------------------

/*! insert_task with the universal chunked-tile fallback.
 *
 * UNWIRED (user order, 2026-10-28): its last caller — transpose's
 * degenerate-geometry valve — now fails loudly instead of staging
 * (chunk classes are powers of two with a 1 MiB floor, so that
 * geometry cannot occur in production). No driver calls this anymore;
 * every chunked op is pointer math and only gemm stages, through its
 * own workspace. The machinery and its counters stay so the probe
 * tests can assert the counter never moves.
 *
 * If no buffer's handle is chunked, this is exactly insert_task. If any
 * is, the op runs through staging: an unchunked scratch Handle (same
 * nbytes) is created for each distinct chunked handle; the converted
 * nntile copy op gathers chunked->scratch for buffers the task reads;
 * the ORIGINAL codelet task is inserted on (scratch, original) handles;
 * the copy op scatters scratch->chunked for buffers the task writes.
 * NNHaul executes tasks on one in-order stream in submit order, so
 * gather->op->scatter ordering is guaranteed by construction.
 *
 * Returns true when the staged path ran (false for the direct path).
 * */
inline bool staged_insert_task(
    ::nnhaul::Codelet &codelet,
    int worker,
    std::vector<BufSpec> const &bufs,
    void const *cl_args,
    std::size_t cl_arg_size);

//! insert_values with the same universal chunked-tile fallback.
inline void staged_insert_values(
    ::nnhaul::Codelet &codelet,
    int worker,
    std::vector<ValSpec> const &vals,
    std::vector<BufSpec> const &bufs);

// ---- implementation -----------------------------------------------------

namespace detail
{

//! One distinct chunked handle of the staged op.
struct StagedBuf
{
    ::nnhaul::Handle *orig = nullptr;
    std::unique_ptr<::nnhaul::Handle> scratch;
    bool gather = false; //!< Copy chunked->scratch before the op.
    bool scatter = false; //!< Copy scratch->chunked after the op
                          //!< (W/RW buffers only; a pure-Read buffer
                          //!< never scatters - see the scatter loop).
};

//! Stage `bufs` around `insert_op` (see staged_insert_task). Returns
//! true when the staged path ran.
template<typename InsertOp>
bool stage_buffers(
    std::vector<BufSpec> const &bufs,
    InsertOp &&insert_op)
{
    bool any_chunked = false;
    for (BufSpec const &buf : bufs)
    {
        if (buf.handle != nullptr && handle_is_chunked(*buf.handle))
        {
            any_chunked = true;
            break;
        }
    }
    if (!any_chunked)
    {
        insert_op(bufs);
        return false;
    }
    // Plan one scratch per distinct chunked handle; merge access modes.
    std::vector<StagedBuf> staged;
    auto find = [&](::nnhaul::Handle *h) -> std::size_t
    {
        for (std::size_t i = 0; i < staged.size(); ++i)
        {
            if (staged[i].orig == h)
            {
                return i;
            }
        }
        return static_cast<std::size_t>(-1);
    };
    for (BufSpec const &buf : bufs)
    {
        if (buf.handle == nullptr || !handle_is_chunked(*buf.handle))
        {
            continue;
        }
        ::nnhaul::Access const access = access_from_mode(buf.mode);
        std::size_t at = find(buf.handle);
        if (at == static_cast<std::size_t>(-1))
        {
            StagedBuf s;
            s.orig = buf.handle;
            // Unchunked scratch of the same payload size; offloadable by
            // default so cap pressure can evict it between uses.
            s.scratch = std::make_unique<::nnhaul::Handle>(
                buf.handle->nbytes());
            staged.push_back(std::move(s));
            at = staged.size() - 1;
        }
        StagedBuf &s = staged[at];
        if (access == ::nnhaul::Access::Read)
        {
            // Pure-Read (so far): the op only reads the scratch and the
            // gather left the orig bytes untouched, so no scatter is
            // needed - writing the identical bytes back would bump the
            // orig's version (a Write), invalidate its host shadow, and
            // make the next eviction re-offload and the next use
            // re-fetch bytes that never changed. scatter stays false; a
            // later W/RW entry for the same handle flips it on below.
            s.gather = true;
        }
        else
        {
            // Write or ReadWrite: the op writes the buffer, so the
            // scatter must land its results back. Gather for Write TOO:
            // many W-mode ops write only a subset of the buffer's bytes
            // (randn's strided output, embedding's indexed rows);
            // without a gather the scatter would overwrite the untouched
            // bytes with the scratch's uninitialized content.
            s.scatter = true;
            s.gather = true;
        }
    }
    // Gather: chunked -> contiguous scratch (the converted copy op walks
    // flat pieces across both chunk tables, so this is one task).
    for (StagedBuf &s : staged)
    {
        if (!s.gather)
        {
            continue;
        }
        try
        {
            ::nntile::haul::copy.submit(-1, *s.orig, *s.scratch);
        }
        catch (::nnhaul::Error const &err)
        {
            std::string const what(err.what());
            if (what.find("never been written") == std::string::npos)
            {
                throw;
            }
            // The one legal Write target that cannot be gathered is a
            // never-written handle (version 0): its bytes are undefined
            // anyway, so skip the gather and let the scatter define
            // them (the op's writes alone land, like the unstaged op).
            continue;
        }
    }
    // The original codelet on (scratch, original) handles.
    std::vector<BufSpec> staged_bufs = bufs;
    for (BufSpec &buf : staged_bufs)
    {
        if (buf.handle == nullptr || !handle_is_chunked(*buf.handle))
        {
            continue;
        }
        buf.handle = staged[find(buf.handle)].scratch.get();
    }
    insert_op(staged_bufs);
    // Scatter: contiguous scratch -> chunked (skipped for pure-Read
    // buffers - the op never wrote the scratch, so the orig bytes are
    // still the ones the gather read and a write-back would only churn
    // the orig's version; a Write buffer whose gather was skipped still
    // scatters - it is the defining write).
    for (StagedBuf &s : staged)
    {
        if (s.scatter)
        {
            note_staged_scatter();
            ::nntile::haul::copy.submit(-1, *s.scratch, *s.orig);
        }
        else
        {
            note_staged_read_skip();
        }
    }
    // Retire each scratch AT ITS LAST CONSUMER (piece H): its last
    // queued task is the scatter for W/RW buffers and the ORIGINAL OP
    // for pure-Read buffers (the skipped scatter no longer trails the
    // op), so its bytes are dead data the moment that task reads them
    // device-side. invalidate() is the non-blocking public marker -
    // nnhaul places it exactly after the last queued task and then
    // reclaims the replica TRANSFERLESS (the dead-handle free; only
    // pre-marker readers could still need the bytes, and they read the
    // resident replica). Without the marker the scratch squats in the
    // victim list until the next wait()-idle drain: under cap pressure
    // the evictor - which cannot know no future task will name the
    // handle - D2H-copied every such squat once (measured on GPT-2
    // classic, cap 2560: 226 GiB of the 340 GiB chunked D2H total was
    // scratch evictions with a matching H2D count of zero - bytes the
    // GPU never asked for again), and the lost cap share pushed real
    // tiles into 2.3x control's eviction volume. The Handle object
    // itself still rides the stash to the wait()-idle drain: the
    // destructor sees dead already set, waits for the (already done or
    // in-flight) sweep, and frees the host shadow page there.
    for (StagedBuf &s : staged)
    {
        s.scratch->invalidate();
    }
    std::vector<std::unique_ptr<::nnhaul::Handle>> scratches;
    scratches.reserve(staged.size());
    for (StagedBuf &s : staged)
    {
        scratches.push_back(std::move(s.scratch));
    }
    stash_scratches(std::move(scratches));
    note_staged_submit();
    return true;
}

} // namespace detail

inline bool staged_insert_task(
    ::nnhaul::Codelet &codelet,
    int worker,
    std::vector<BufSpec> const &bufs,
    void const *cl_args,
    std::size_t cl_arg_size)
{
    return detail::stage_buffers(
        bufs,
        [&](std::vector<BufSpec> const &task_bufs)
        {
            insert_task(codelet, worker, task_bufs, cl_args, cl_arg_size);
        });
}

inline void staged_insert_values(
    ::nnhaul::Codelet &codelet,
    int worker,
    std::vector<ValSpec> const &vals,
    std::vector<BufSpec> const &bufs)
{
    detail::stage_buffers(
        bufs,
        [&](std::vector<BufSpec> const &task_bufs)
        {
            insert_values(codelet, worker, vals, task_bufs);
        });
}

} // namespace stage
} // namespace haul
} // namespace nntile

#endif // NNTILE_USE_NNHAUL
