/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nnhaul/chunk_units.hh
 * Shared building blocks for NATIVE chunked drivers of unit-structured
 * classic ops (docs/chunked-tiles.md §4 item 3): the chunk segmentation
 * walker, the unit-run enumerator and the straddle staging arenas, in the
 * host+device discipline established by src/nnhaul/gemm.cc and
 * src/nnhaul/embedding_backward.cc.
 *
 * @version 1.2.0
 * */

#pragma once

#ifdef NNTILE_USE_NNHAUL

#include <nntile/base_types.hh>
#include <nntile/defs.h>

#include <nnhaul/nnhaul.hh>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#ifdef NNTILE_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nntile::haul::chunkunits
{

//! Compile-time switch (NNTILE_FORCE_CHUNKED_KERNELS): every op dispatches
//! through its chunked kernel even when the operands are contiguous. An
//! unchunked buffer is the one-chunk degenerate case, so the chunked kernel
//! runs with nchunks=1 and the result is bitwise the classic kernel's.
inline constexpr bool force_chunked_kernels() noexcept
{
#ifdef NNTILE_FORCE_CHUNKED_KERNELS
    return true;
#else
    return false;
#endif
}

//! The ONE chunk class (rule 7): equals nnhaul::kChunkClassBytes, a
//! compile-time power of two. A multi-chunk buffer's full chunk is
//! exactly the class -- the nnhaul policy (chunk_bytes_for) cannot
//! create anything else -- and ChunkView aborts loudly on a violation.
//! The retired virtual-rechunk test hook (NNHAUL_CHUNK_UNITS_FORCE_BYTES,
//! gemm's NNHAUL_GEMM_FORCE_PANEL_BYTES) died with the configurable
//! class: a hand-forced non-class factorization is now an invariant
//! violation, and the checks run the REAL 16 MiB factorizations.
inline constexpr std::size_t kChunkClassBytes = ::nnhaul::kChunkClassBytes;

//! Classic-dispatch tripwire (NNTILE_FORCE_CHUNKED_KERNELS): under the
//! flag, dispatching a classic contiguous kernel is a wiring bug -- every
//! op must go through its chunked kernel, and the flag-on training run
//! itself then proves zero classic dispatches by surviving. (rule 11,
//! 2026-10-19: the structural-overflow ring fallthrough no longer exists
//! -- a launch the ring cannot serve aborts; every wrapper passes false.)
//! Unconditional: no env gate. Compiles to nothing when the flag is off.
inline void classic_dispatch_tripwire(char const *op, bool ring_overflow)
    noexcept
{
#ifdef NNTILE_FORCE_CHUNKED_KERNELS
    if(!ring_overflow)
    {
        std::fprintf(stderr,
            "nntile %s: classic kernel dispatch under "
            "NNTILE_FORCE_CHUNKED_KERNELS\n",
            op);
        std::abort();
    }
#else
    (void)op;
    (void)ring_overflow;
#endif
}

//! Chunk residency of one buffer's flat element space: equal-size chunks
//! with a shorter tail; an unchunked buffer is the 1-chunk degenerate case
//! (same arithmetic as the anonymous ChunkView of src/nnhaul/gemm.cc,
//! published here so the unit-structured families share one walker).
//! Rule 7: a MULTI-CHUNK buffer's full chunk is exactly the compile-time
//! class (kChunkClassBytes, a power of two) — anything else is an
//! invariant violation and aborts loudly (the nnhaul policy cannot
//! create it; only a hand-built Handle could). A 1-chunk buffer keeps
//! chunk_elems = its total (the host-side walkers' boundary logic sees
//! the buffer end as the chunk end); the DEVICE kernels never divmod a
//! 1-chunk operand at all (their shift is the single-chunk sentinel).
template<typename T>
struct ChunkView
{
    ::nnhaul::Buffer *buf = nullptr;
    std::size_t chunk_elems = 1; //!< elements of a full chunk
    std::size_t nchunks = 1;
    std::size_t total_elems = 0;
    bool real = false; //!< true: per-chunk pointers; false: one block

    ChunkView() = default;
    explicit ChunkView(::nnhaul::Buffer *b):
        buf(b),
        total_elems(b->nbytes() / sizeof(T)),
        nchunks(b->chunk_count())
    {
        if(nchunks > 1)
        {
            if(b->chunk_bytes(0) != kChunkClassBytes)
            {
                std::fprintf(stderr,
                    "nntile ChunkView: multi-chunk buffer %zu-byte full "
                    "chunk != the compile-time %zu-byte class (rule 7)\n",
                    b->chunk_bytes(0),
                    kChunkClassBytes);
                std::abort();
            }
            chunk_elems = kChunkClassBytes / sizeof(T);
            real = true;
        }
        else
        {
            chunk_elems = total_elems > 0 ? total_elems : 1;
            nchunks = 1;
            real = false;
        }
    }

    std::size_t chunk_of(std::size_t elem) const noexcept
    {
        return elem / chunk_elems;
    }
    std::size_t chunk_begin(std::size_t c) const noexcept
    {
        return c * chunk_elems;
    }
    std::size_t chunk_end(std::size_t c) const noexcept
    {
        std::size_t const e = (c + 1) * chunk_elems;
        return e < total_elems ? e : total_elems;
    }
    //! Typed pointer at chunk c, off elements into that chunk.
    T *ptr(std::size_t c, std::size_t off) const
    {
        if(real)
        {
            return buf->get_chunk_ptr<T>(c) + off;
        }
        return buf->get_ptr<T>() + chunk_begin(c) + off;
    }
};

//! True when the flat element range [f, f+len) lies inside one chunk.
template<typename T>
bool in_one_chunk(ChunkView<T> const &cv, std::size_t f, std::size_t len)
    noexcept
{
    if(len == 0)
    {
        return true;
    }
    std::size_t const c = cv.chunk_of(f);
    return f + len <= cv.chunk_end(c);
}

//! Host-memory scratch arena for CPU bodies.
template<typename T>
struct ArenaCpu
{
    std::vector<std::vector<T>> blocks;

    T *alloc(std::size_t elems)
    {
        blocks.emplace_back(elems == 0 ? 1 : elems);
        return blocks.back().data();
    }
    //! copy/zero are element-type-polymorphic so a driver that mixes
    //! operand types (e.g. mask_scalar's bool mask beside fp data) can
    //! stage the other type through the same arena (a U* temp reinterpreted
    //! from an over-allocated T block).
    template<typename U>
    void copy(U *dst, U const *src, std::size_t elems) noexcept
    {
        if(elems > 0)
        {
            std::memcpy(dst, src, elems * sizeof(U));
        }
    }
    template<typename U>
    void zero(U *dst, std::size_t elems) noexcept
    {
        if(elems > 0)
        {
            std::memset(dst, 0, elems * sizeof(U));
        }
    }
    bool ok_state() const noexcept
    {
        return true;
    }
};

#ifdef NNTILE_USE_CUDA
//! Stream-ordered device scratch arena for CUDA bodies: every block is
//! cudaMallocAsync'd on the compute stream and freed (cudaFreeAsync) at
//! scope end, so all gather/kernel/scatter work enqueued in between is
//! stream-ordered before the release. Error convention as in gemm.cc:
//! report on stderr, flip ok, the driver stops issuing further calls and
//! the result stays partially unwritten (same failure class as a failed
//! kernel launch).
template<typename T>
struct ArenaCuda
{
    cudaStream_t stream;
    std::vector<T *> blocks;
    bool ok = true;

    explicit ArenaCuda(cudaStream_t s): stream(s)
    {
    }
    T *alloc(std::size_t elems)
    {
        T *p = nullptr;
        cudaError_t const err = cudaMallocAsync(
            &p, (elems == 0 ? 1 : elems) * sizeof(T), stream);
        if(err != cudaSuccess)
        {
            std::fprintf(stderr,
                "nntile chunk_units: cudaMallocAsync failed: %s\n",
                cudaGetErrorString(err));
            ok = false;
            return nullptr;
        }
        blocks.push_back(p);
        return p;
    }
    //! Element-type-polymorphic like ArenaCpu::copy (mixed-type staging).
    template<typename U>
    void copy(U *dst, U const *src, std::size_t elems) noexcept
    {
        if(elems == 0 || !ok)
        {
            return;
        }
        cudaError_t const err = cudaMemcpyAsync(dst, src, elems * sizeof(U),
            cudaMemcpyDeviceToDevice, stream);
        if(err != cudaSuccess)
        {
            std::fprintf(stderr,
                "nntile chunk_units: cudaMemcpyAsync failed: %s\n",
                cudaGetErrorString(err));
            ok = false;
        }
    }
    template<typename U>
    void zero(U *dst, std::size_t elems) noexcept
    {
        if(elems == 0 || !ok)
        {
            return;
        }
        cudaError_t const err = cudaMemsetAsync(dst, 0, elems * sizeof(U),
            stream);
        if(err != cudaSuccess)
        {
            std::fprintf(stderr,
                "nntile chunk_units: cudaMemsetAsync failed: %s\n",
                cudaGetErrorString(err));
            ok = false;
        }
    }
    bool ok_state() const noexcept
    {
        return ok;
    }
    ~ArenaCuda()
    {
        for(T *p : blocks)
        {
            cudaError_t const err = cudaFreeAsync(p, stream);
            if(err != cudaSuccess)
            {
                std::fprintf(stderr,
                    "nntile chunk_units: cudaFreeAsync failed: %s\n",
                    cudaGetErrorString(err));
            }
        }
    }
};

//! Cyclic DEVICE ring of chunk base-pointer slots for the coarse-chunk
//! transport (docs/chunked-tiles.md §2/§4). This replaces the earlier
//! host-mapped arena, which was a correctness hazard: a kernel read the
//! pinned host array AT EXECUTION TIME, and with the driver running ahead
//! on the in-order compute stream a queued-but-unexecuted kernel could
//! read slots the host had already repopulated for a later launch (a tile
//! offloaded and prefetched between two uses changes its chunk addresses,
//! so the earlier kernel would dereference the later use's pointers).
//!
//! The ring lives in device memory, is read-only to kernels, and is
//! written BEFORE the launch with a stream-ordered `cudaMemcpyAsync` on
//! the SAME compute stream the kernel runs on. Stream order then makes
//! reuse safe by construction: a later upload into a slot executes after
//! every earlier kernel on that stream (so the earlier kernel has already
//! read the slot). The upload is enqueued immediately before the kernel
//! and there is no cross-stream dependency: the data manager's
//! offload/prefetch work rides its own copy streams and never touches the
//! ring, only the chunk data blocks the kernel reads.
//!
//! Per launch, each operand's `n` chunk pointers are appended at the
//! cursor (exactly `n` slots -- nothing is reserved per tile) and the
//! kernel is handed the DEVICE pointer `ring + start` plus its count, i.e.
//! a `T const* const*` view, exactly as before. A tile with any number of
//! chunks simply needs that many consecutive slots; there is no per-tile
//! or in-flight-handle cap. The only bound is the ring's free-slot count:
//! when a launch's pointers do not fit in the remaining space, the ring
//! waits for the most recent recorded launch to complete (its event, in
//! stream order, is after every earlier kernel) and wraps the cursor to 0.
//! The one structural case the ring skips is a single launch whose pointer
//! set exceeds kChunkPtrRingSlots even when empty: `upload` returns
//! nullptr -- rule 11 (2026-10-19): callers ABORT; no per-piece driver
//! exists anywhere.
//! (piece E, user directives 2026-10-19) The ring is ONE ring on each
//! side of the bus: the device ring (kChunkPtrRingSlots pow2 slots, the
//! historical transport) and a PINNED HOST MIRROR with the SAME slot
//! indexing, sharing ONE cursor, ONE wrap event and ONE wrap. A
//! launch builds its merged table straight into the host mirror at the
//! cursor; the upload is ONE `cudaMemcpyAsync` host+base -> dev+base on
//! the dedicated CONTROL stream, ordered ahead of the kernel by an
//! EVENT (record + compute-stream wait) -- the host never blocks and
//! mirror slots stay append-only until the wrap. Device slots are
//! append-only until a wrap too, and the wrap
//! synchronizes the compute-stream event recorded after the latest
//! table-reading kernel (end()) first -- so the full 2^20-slot depth of
//! the ring is planning room for in-flight launches (kernel 66, or
//! kernel 100066, launches whenever total not-yet-copied table slots
//! fit the ring; no per-launch or 64-launch buffering exists, and every
//! launch uploads its table -- 8 bytes per 16 MB chunk is noise).
//!
//! The malloc doctrine (user directive): the mirror is one real
//! cudaHostAlloc of the ring's slot bytes; if it fails, the transport
//! cannot exist and the error is loud -- there is no constant-derived
//! bound anywhere. A single launch whose table exceeds the whole ring
//! (more chunk slots than the pow2 ring holds) is an invariant
//! violation (rule 11): the wrapper aborts.
struct RingTables;

struct ChunkPtrRing
{
    /*! Ring capacity in pointer slots (8 MiB at 8 bytes per slot).
     * INVARIANT (user directive, 2026-10-18): the slot capacity is a
     * COMPILE-TIME POWER OF TWO, and every effective capacity (the test
     * hook below may only shrink it to another power of two) stays a
     * power of two -- the cursor/wrap arithmetic may rely on it. This is
     * the ring's slot capacity only; per-tile chunk counts stay exactly
     * what they are (per-tile, arbitrary, tails included). */
    static constexpr std::size_t kChunkPtrRingSlots = std::size_t{1} << 20;
    static constexpr std::size_t kChunkPtrBytes = sizeof(void *);
    static constexpr std::size_t kChunkPtrRingBytes =
        kChunkPtrRingSlots * kChunkPtrBytes;

    char *dev = nullptr;
    void **host = nullptr; //!< pinned mirror, same slot indexing as dev
    std::size_t cursor = 0; //!< next free slot (host and dev in lockstep)
    std::size_t capacity = kChunkPtrRingSlots; //!< slots (test hook may shrink)
    std::size_t wraps = 0; //!< reclamations performed (diagnostic)
    cudaStream_t stream = nullptr; //!< the kernel execution (compute) stream
    /*! (user directives 2026-10-19; event-ordered revision after the
     * pair-1 measurement) The CONTROL stream: chunk-pointer H2D rides
     * its own stream so a table copy never queues behind the previous
     * kernel's EXECUTION on the compute stream, and the kernel's
     * device timeline carries no CE gap in front of it. Completion is
     * ordered by an EVENT, not by host blocking: the copy is enqueued,
     * `ready_ev` is recorded behind it on the control stream, and the
     * compute stream waits on that event before the kernel launch --
     * the kernel still cannot start before its table is resident, but
     * the worker thread never stalls (pair-1 measured the blocking
     * form serializing the launch-ahead pipeline: idle-before-multi-
     * chunk gaps 689 us vs 354-501 in every prior traced run, ~0.7 s
     * per traced loop). Host-side table building also proceeds into
     * the append-only mirror while prior kernels fly; only the rare
     * build-time wrap still synchronizes. */
    cudaStream_t control = nullptr;
    cudaEvent_t ready_ev = nullptr; //!< fires when the latest table copy lands
    cudaEvent_t wrap_ev = nullptr; //!< behind the latest table-reading kernel
    bool ev_recorded = false;

    /*! Test hook: NNHAUL_CHUNK_RING_SLOTS shrinks the ring so the wrap/
     * reclaim path can be exercised on hardware whose VRAM cannot fill
     * the default 2^20-slot ring. POW2 ONLY (the capacity invariant
     * above): a non-power-of-two value warns once and is ignored, so
     * the invariant cannot be violated even under test. Default stays
     * kChunkPtrRingSlots. Read ONCE per process (piece 3): the stress
     * checks set it at main() start. */
    std::size_t ring_capacity() const noexcept
    {
        static std::size_t const cached = [] {
            char const *e = std::getenv("NNHAUL_CHUNK_RING_SLOTS");
            if(e != nullptr)
            {
                long v = std::atol(e);
                if(v > 0
                    && (static_cast<std::size_t>(v)
                            & (static_cast<std::size_t>(v) - 1))
                        == 0)
                {
                    return static_cast<std::size_t>(v);
                }
                std::fprintf(stderr,
                    "nntile chunk_units: NNHAUL_CHUNK_RING_SLOTS=%s is not "
                    "a positive power of two; keeping %zu slots\n",
                    e,
                    kChunkPtrRingSlots);
            }
            return kChunkPtrRingSlots;
        }();
        return cached;
    }

    /*! The pinned host mirror (allocated once, on the first launch
     * build, through the dynamic malloc mechanism: success = the
     * transport exists; failure is a loud error, not a fallback). */
    bool ensure_host()
    {
        capacity = ring_capacity();
        if(host != nullptr)
        {
            return true;
        }
        if(cudaHostAlloc(reinterpret_cast<void **>(&host),
            capacity * kChunkPtrBytes, cudaHostAllocDefault)
            != cudaSuccess)
        {
            host = nullptr;
            std::fprintf(stderr,
                "nntile chunk_units: host-mirror cudaHostAlloc(%zu) failed; "
                "the chunk transport cannot exist\n",
                capacity * kChunkPtrBytes);
            return false;
        }
        return true;
    }

    //! Ensure the ring exists and remember this instance's compute stream.
    //! Returns false on allocation failure (the caller aborts: rule 11).
    bool begin(cudaStream_t s)
    {
        stream = s;
        capacity = ring_capacity();
        if(dev == nullptr)
        {
            if(cudaMalloc(reinterpret_cast<void **>(&dev),
                kChunkPtrRingBytes) != cudaSuccess)
            {
                std::fprintf(stderr, "nntile chunk_units: chunk-ptr ring "
                    "cudaMalloc(%zu) failed\n", kChunkPtrRingBytes);
                return false;
            }
        }
        if(wrap_ev == nullptr)
        {
            // (critic C6, 2026-10-19) FATAL: wrap_now() rewinds the
            // cursor under the assumption this event exists -- without
            // it a wrap would silently overwrite slots a queued kernel
            // may still read.
            if(cudaEventCreateWithFlags(&wrap_ev, cudaEventDisableTiming)
                != cudaSuccess)
            {
                std::fprintf(stderr, "nntile chunk_units: wrap event "
                    "create failed; refusing to run without wrap "
                    "safety\n");
                return false;
            }
        }
        if(control == nullptr &&
            cudaStreamCreate(&control) != cudaSuccess)
        {
            control = nullptr;
            std::fprintf(stderr, "nntile chunk_units: control stream "
                "create failed\n");
            return false;
        }
        if(ready_ev == nullptr &&
            cudaEventCreateWithFlags(&ready_ev, cudaEventDisableTiming)
                != cudaSuccess)
        {
            ready_ev = nullptr;
            std::fprintf(stderr, "nntile chunk_units: ready event "
                "create failed\n");
            return false;
        }
        return true;
    }

    /*! Wrap NOW: every H2D copy that reads slots [0, cursor) is ordered
     * before the recorded copy event -- synchronize it, then rewind.
     * Called at BUILD time by RingTables::reserve_more when a table
     * would cross the physical end (the table must stay contiguous for
     * its one memcpy and its kernel's linear indexing). */
    void wrap_now()
    {
        if(ev_recorded && wrap_ev != nullptr)
        {
            cudaEventSynchronize(wrap_ev);
        }
        cursor = 0;
        ++wraps;
    }

    /*! Core of the legacy single-table form (the ring stress check
     * drives this one directly; drivers use the merged upload_tables).
     * Wrap/reclaim and slot accounting as ever; the source is the
     * caller's buffer. */
    void const *upload_bytes(void const *src, std::size_t n)
    {
        if(dev == nullptr || n == 0 || n > capacity)
        {
            return nullptr;
        }
        if(cursor + n > capacity)
        {
            wrap_now();
        }
        std::size_t const off = cursor;
        if(cudaMemcpyAsync(dev + off * kChunkPtrBytes, src,
            n * kChunkPtrBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        {
            std::fprintf(stderr, "nntile chunk_units: chunk-ptr ring "
                "cudaMemcpyAsync failed\n");
            return nullptr;
        }
        cursor += n;
        if(wrap_ev != nullptr)
        {
            cudaEventRecord(wrap_ev, stream);
            ev_recorded = true;
        }
        return dev + off * kChunkPtrBytes;
    }

    //! Append `ptrs` at the cursor and return the device view the kernel
    //! reads (nullptr when the caller must error out). Legacy form.
    template<typename P>
    P const *upload(std::vector<P> const &ptrs)
    {
        static_assert(sizeof(P) == kChunkPtrBytes,
            "ring slots are pointer-sized");
        return reinterpret_cast<P const *>(
            upload_bytes(ptrs.data(), ptrs.size()));
    }

    /*! (piece E) ONE merged upload per launch: the operand tables of
     * the whole launch, built by RingTables into the host mirror at the
     * shared cursor, ride a single stream-ordered H2D copy into the
     * same offset of the device ring. Defined after RingTables (it
     * reads the builder's base/size). */
    void const *upload_tables(RingTables &t);

    //! Typed device view of the operand table that starts at slot offset
    //! `off` of an upload_tables range. P is the SLOT type: `const T *`
    //! for a source table, `T *` for a destination table, `Index` for
    //! the maxsumexp straddling-panel list.
    template<typename P>
    static P const *table_view(void const *base, std::size_t off) noexcept
    {
        static_assert(sizeof(P) == kChunkPtrBytes,
            "ring slots are pointer-sized");
        return reinterpret_cast<P const *>(
            static_cast<char const *>(base) + off * kChunkPtrBytes);
    }

    /*! End-of-launch marker: records, on the COMPUTE stream, the event
     * a future wrap waits on -- with table copies on the (event-ordered)
     * control stream, this event behind the latest table-reading kernel
     * is the only thing standing between a wrap's overwrite and a
     * still-queued reader (stream order finishes every earlier kernel
     * first). */
    void end(cudaStream_t s)
    {
        if(dev != nullptr && wrap_ev != nullptr)
        {
            cudaEventRecord(wrap_ev, s);
            ev_recorded = true;
        }
    }
};

inline ChunkPtrRing &chunk_ptr_ring() noexcept
{
    // Function-local static: one instance across all translation units that
    // include this header.
    static ChunkPtrRing a;
    return a;
}

/*! (piece 3, restructured piece E) Launch-scope builder of ONE merged
 * ring upload: each operand's chunk-pointer table is appended into the
 * SHARED host mirror at the shared cursor (base + n), the whole
 * concatenation rides one cudaMemcpyAsync, and the kernel-visible bytes
 * and slot accounting are exactly the historical merged upload's.
 *
 * The builder keeps only two words of per-launch state: the base slot
 * (captured at clear()) and the running size. A table that would cross
 * the ring's physical end wraps AT BUILD TIME (the ring's one copy
 * event is synchronized, the cursor rewinds, and the few already-written
 * slots are moved to the new base) -- so the uploaded range is always
 * one contiguous slot range and the kernel keeps its linear indexing. */
struct RingTables
{
    static constexpr std::size_t kSlotBytes = sizeof(void *);

    std::size_t base = 0; //!< this launch's first slot in the mirror
    std::size_t n = 0; //!< slots used by the launch being built

    //! Start building this launch's merged table (the mirror must exist;
    //! a failed cudaHostAlloc is a loud error -- there is no fallback).
    void clear() noexcept
    {
        ChunkPtrRing &r = chunk_ptr_ring();
        if(!r.ensure_host())
        {
            std::abort();
        }
        base = r.cursor;
        n = 0;
    }

    //! Room for k more slots of THIS launch, wrapping at build time when
    //! the table would cross the ring's physical end. A table larger
    //! than the WHOLE ring (more chunk slots than the pow2 ring holds)
    //! is an invariant violation (rule 11) and throws.
    void reserve_more(std::size_t k)
    {
        ChunkPtrRing &r = chunk_ptr_ring();
        if(n + k > r.capacity)
        {
            throw ::nnhaul::Error(
                "nntile chunk_units: launch table of "
                + std::to_string(n + k)
                + " slots exceeds the whole chunk-ptr ring ("
                + std::to_string(r.capacity)
                + " slots; rule 11 -- no fallback exists)");
        }
        if(base + n + k <= r.capacity)
        {
            return;
        }
        std::size_t const old_base = base;
        r.wrap_now();
        base = 0;
        if(n > 0 && old_base != 0)
        {
            std::memmove(r.host, r.host + old_base, n * kSlotBytes);
        }
    }

    //! Append one operand's chunk-pointer table and return the slot
    //! offset of its first entry. Duck-typed on the operand view: the
    //! shared ChunkView and the embedding family's local view structs
    //! (EmbChunkView/EmbBwdChunkView) all expose nchunks and ptr(c, 0).
    template<typename View>
    std::size_t add(View const &cv)
    {
        std::size_t const off = n;
        reserve_more(cv.nchunks);
        void **slots = chunk_ptr_ring().host + base;
        for(std::size_t c = 0; c < cv.nchunks; ++c)
        {
            slots[n++] = cv.ptr(c, 0);
        }
        return off;
    }

    //! Append a pre-built table by byte copy (the maxsumexp straddling
    //! panel list rides the same slots; P must be one 8-byte slot).
    template<typename P>
    std::size_t add(std::vector<P> const &ptrs)
    {
        static_assert(sizeof(P) == kSlotBytes,
            "ring slots are pointer-sized");
        std::size_t const off = n;
        reserve_more(ptrs.size());
        if(!ptrs.empty())
        {
            std::memcpy(chunk_ptr_ring().host + base + n, ptrs.data(),
                ptrs.size() * sizeof(P));
            n += ptrs.size();
        }
        return off;
    }

    void const *staging_data() const noexcept
    {
        return chunk_ptr_ring().host + base;
    }
    std::size_t size() const noexcept
    {
        return n;
    }
};

/*! The per-thread launch builder (the words base/n are per-launch; the
 * bytes live in the one shared host mirror). SINGLE-CUDA-WORKER
 * INVARIANT (critic C5, 2026-10-19): the runtime runs exactly one CUDA
 * worker and one compute stream (nnhaul.hh: "it always runs one CUDA
 * worker"), so this thread-local builder and the shared mirror+cursor
 * cannot interleave. Two CUDA workers would build at the same base and
 * corrupt each other's tables -- the runtime forbids that today; this
 * is the transport's load-bearing assumption, stated here. */
inline RingTables &ring_tables() noexcept
{
    thread_local RingTables t;
    return t;
}

/*! (piece E final form, user directive 2026-10-19: NO cache) The
 * merged upload: ONE cudaMemcpyAsync of the launch's table from the
 * pinned host mirror into the same offset of the device ring, ON THE
 * CONTROL STREAM, ordered ahead of the kernel by cudaEventRecord +
 * cudaStreamWaitEvent -- the kernel cannot start before its table is
 * resident and the host never blocks; mirror slots stay append-only
 * until the build-time wrap. The cursor advances past
 * the table. Every launch uploads its pointers -- 8 bytes per 16 MB
 * chunk is noise -- and the 2^20-slot ring is the planning depth:
 * device slots are append-only until the (rare) build-time wrap
 * synchronizes end()'s compute-stream event (the last table-reading
 * kernel). A launch larger than the whole ring throws at build time
 * (rule 11). */
inline void const *ChunkPtrRing::upload_tables(RingTables &t)
{
    std::size_t const n = t.size();
    if(n == 0 || n > capacity || dev == nullptr || control == nullptr
        || ready_ev == nullptr)
    {
        return nullptr;
    }
    if(cudaMemcpyAsync(dev + t.base * kChunkPtrBytes, t.staging_data(),
        n * kChunkPtrBytes, cudaMemcpyHostToDevice, control) != cudaSuccess
        || cudaEventRecord(ready_ev, control) != cudaSuccess
        || cudaStreamWaitEvent(stream, ready_ev, 0) != cudaSuccess)
    {
        std::fprintf(stderr, "nntile chunk_units: chunk-ptr table control "
            "copy/ordered-launch failed\n");
        return nullptr;
    }
    // The kernel enqueued next on the compute stream is ordered behind
    // the copy by ready_ev: it cannot start until its table is resident,
    // and the host never blocked. The mirror slots this table was built
    // from are reused only after the wrap sync (append-only until then).
    cursor = t.base + n;
    return dev + t.base * kChunkPtrBytes;
}

#endif // NNTILE_USE_CUDA

//! Gather the flat element range [start, start+len) of a buffer into one
//! contiguous scratch run (mirror: scatter it back).
template<typename T, typename Arena>
void gather_range(ChunkView<T> const &cv, std::size_t start,
    std::size_t len, T *dst, Arena &arena) noexcept
{
    std::size_t done = 0;
    while(done < len)
    {
        std::size_t const at = start + done;
        std::size_t const c = cv.chunk_of(at);
        std::size_t n = cv.chunk_end(c) - at;
        if(n > len - done)
        {
            n = len - done;
        }
        arena.copy(dst + done, cv.ptr(c, at - cv.chunk_begin(c)), n);
        done += n;
    }
}

template<typename T, typename Arena>
void scatter_range(ChunkView<T> const &cv, std::size_t start,
    std::size_t len, T const *src, Arena &arena) noexcept
{
    std::size_t done = 0;
    while(done < len)
    {
        std::size_t const at = start + done;
        std::size_t const c = cv.chunk_of(at);
        std::size_t n = cv.chunk_end(c) - at;
        if(n > len - done)
        {
            n = len - done;
        }
        arena.copy(cv.ptr(c, at - cv.chunk_begin(c)), src + done, n);
        done += n;
    }
}

/*! One unit-structured operand view over the shared row index space.
 *
 * The softmax-class and slice kernels (kernel/softmax*, maxsumexp,
 * logsumexp, *_slice) all address their m-by-k-by-n operands as
 * kernel-derived flat maps with the CONTIGUOUS axis of extent m first:
 * element (i0, i1, i2) sits at flat i0 + m*i1 + m*k*i2 (see the layout
 * derivation in the driver comments). The natural decomposition unit is
 * therefore the ROW of m contiguous elements, and the natural reduction
 * group is the PANEL of k consecutive rows whose fibers share one
 * m-element (softmax-class: 2*m-element) slice segment.
 *
 * A UnitSpan tracks one operand's residency in that index space:
 *  - a ROW span (unit_rows = 1, unit = m): unit r covers flat elements
 *    [r*m, (r+1)*m); the walker never splits a row, staging rows that
 *    straddle a chunk boundary through a bounded m-element temp;
 *  - a PANEL span (unit_rows = k, unit = seg): unit p covers the slice
 *    segment [p*seg, (p+1)*seg); a segment that straddles a chunk
 *    boundary is staged whole (bounded seg elements) and forces
 *    single-panel pieces.
 * */
template<typename T>
struct UnitSpan
{
    ChunkView<T> const *cv = nullptr;
    std::size_t unit = 0; //!< elements per unit (row: m; panel: seg)
    Index unit_rows = 1; //!< rows per unit (row: 1; panel: k)

    //! Flat element offset of unit u.
    std::size_t offset(Index u) const noexcept
    {
        return static_cast<std::size_t>(u) * unit;
    }

    //! True when the unit containing row r crosses a chunk boundary.
    bool straddles(Index r) const noexcept
    {
        if(unit == 0)
        {
            return false;
        }
        return !in_one_chunk(*cv, offset(r / unit_rows), unit);
    }

    //! Whole units from the unit of row r to its chunk end, in ROWS
    //! (>= 1 row for a non-straddling row unit; 0 for a panel unit
    //! whose segment straddles).
    Index run(Index r) const noexcept
    {
        Index const u = r / unit_rows;
        if(unit == 0)
        {
            return std::numeric_limits<Index>::max();
        }
        std::size_t const s = offset(u);
        std::size_t const c = cv->chunk_of(s);
        Index n = static_cast<Index>((cv->chunk_end(c) - s) / unit);
        return n * unit_rows;
    }

    //! Pointer to the unit of row r (valid only when !straddles(r)).
    T *ptr(Index r) const
    {
        Index const u = r / unit_rows;
        std::size_t const s = offset(u);
        std::size_t const c = cv->chunk_of(s);
        return cv->ptr(c, s - cv->chunk_begin(c));
    }
};

/*! Enumerate the unit runs of a unit-structured op over its row index
 * space [0, n_panels * panel_rows):
 *
 *  - pieces are maximal runs of whole rows that each lie inside single
 *    chunks of every ROW span; a row straddling any row span becomes a
 *    len-1 piece with the corresponding flag set (the caller stages that
 *    row through its bounded temp);
 *  - a piece starting mid-panel is confined to that panel (its kernel
 *    sub-call is (m, 1, len) over the panel's slice segment);
 *  - a piece starting at a panel boundary may span whole panels only
 *    ((m, n', k) sub-calls), capped so that every touched PANEL segment
 *    lies inside single chunks; a straddling panel segment (flag set)
 *    confines the piece to one panel and the caller stages the segment.
 *
 * The callback receives (begin_row, len_rows, first_in_panel, flags...):
 * one straddle flag per span, in order - row spans flag straddling rows,
 * panel spans flag the straddling segment of the piece's first panel.
 * first_in_panel is true when the piece starts exactly at a panel
 * boundary (the reduction drivers' beta-chaining anchor: op beta for
 * first pieces, 1 for continuations).
 *
 * Degenerate shapes (panel_rows == 0) emit nothing.
 * */
template<typename Emit, typename... Ts>
void walk_unit_runs(Index n_panels, Index panel_rows, Emit &&emit,
    UnitSpan<Ts> const &...spans)
{
    if(n_panels <= 0 || panel_rows <= 0)
    {
        return;
    }
    Index const rows = n_panels * panel_rows;
    Index i = 0;
    while(i < rows)
    {
        bool const any_row_straddle = (spans.straddles(i) || ...);
        if(any_row_straddle)
        {
            Index const panel = i / panel_rows;
            emit(i, Index{1}, i % panel_rows == 0, spans.straddles(i)...);
            ++i;
            continue;
        }
        Index len = rows - i;
        ((len = len < spans.run(i) ? len : spans.run(i)), ...);
        if(i % panel_rows != 0)
        {
            // Mid-panel head/tail: confine to the current panel.
            Index const in_panel = panel_rows - i % panel_rows;
            len = len < in_panel ? len : in_panel;
        }
        else
        {
            // Panel-aligned: whole panels only, capped by the panel
            // segments' single-chunk residency (P == 0 means some panel
            // segment straddles - fall back to a single staged panel).
            Index panels = n_panels - i / panel_rows;
            ((panels = panels < spans.run(i) / panel_rows
                     ? panels
                     : spans.run(i) / panel_rows),
                ...);
            if(panels >= 1)
            {
                Index const capped = panels * panel_rows;
                len = len < capped ? len : capped;
                len = len / panel_rows * panel_rows;
            }
            else
            {
                len = len < panel_rows ? len : panel_rows;
            }
        }
        if(len <= 0)
        {
            // A row span's run() can hit 0 only together with a straddle
            // (handled above); guard anyway to make progress.
            len = 1;
        }
        emit(i, len, i % panel_rows == 0, spans.straddles(i)...);
        i += len;
    }
}

/*! Split one piece of the fiber-class row space into kernel-expressible
 * sub-pieces (the family-2 fiber promotion, cf. the slice-class walker
 * above).
 *
 * Fiber index map (derived from kernel/{add,scale,multiply,sum,norm,
 * sumprod}_fiber* cpu/cuda, NOT assumed): the tensor operands are
 * m-by-k-by-n(-by-batch) with element (i0, i2, i1, b) at flat
 * i0 + m*(i2 + k*(i1 + n*b)) - the CONTIGUOUS unit is the m-element
 * fiber along i0 (the first tensor axis), fibers group into k-fiber
 * PLANES per (i1, b), and the small fiber operand (k*batch or k
 * elements) is indexed by (i2, b): it re-bases at every batch boundary
 * and varies WITHIN a plane along the contiguous fiber index. Unlike
 * the slice class (small operand constant along the contiguous axis,
 * one segment per panel), a run of fibers is a valid kernel sub-tensor
 * only when it does not cross a plane boundary unless it starts
 * plane-aligned, and never crosses a batch boundary:
 *  - a mid-plane sub-piece [i, plane_end) is the sub-call (m, 1, L, 1)
 *    at tensor offset m*i reading the fiber operand at its batch
 *    segment plus offset (i % k);
 *  - a plane-aligned sub-piece of P whole planes inside one batch is
 *    the sub-call (m, P, k, 1) at tensor offset m*i reading the fiber
 *    operand at its batch segment plus 0;
 *  - the fiber operand's batch segment is [k*b, k*(b+1)) - one
 *    SliceStage segment per batch (seg = k) when staging is needed.
 *
 * split_fiber_piece re-cuts the (already chunk-resident) piece
 * [i, i+len) accordingly and emits (start, len, plane_span,
 * first_in_batch) per sub-piece: plane_span is true for a whole-plane
 * run ((m, len/k, k, 1) sub-call), false for a single-plane partial
 * ((m, 1, len, 1) with fiber offset start % k); first_in_batch is true
 * when the sub-piece starts in plane 0 of its batch (start % (k*n) <
 * k) - the reduction chaining anchor: every m-fiber's first
 * contribution comes from its plane-0 row, so a first_in_batch
 * sub-piece takes the caller's (alpha, beta) and every continuation
 * takes (alpha, 1). Sub-pieces of a single-chunk piece stay
 * single-chunk, so the chunk-residency caps of the outer walk hold.
 * */
template<typename Emit>
void split_fiber_piece(Index i, Index len, Index n, Index k, Emit &&emit)
{
    Index const end = i + len;
    while(i < end)
    {
        Index const i2 = i % k;
        bool const first = i % (n * k) < k;
        if(i2 != 0)
        {
            // Mid-plane head: confined to the remainder of its plane.
            Index const to_plane_end = (i / k + 1) * k - i;
            Index const L = to_plane_end < end - i ? to_plane_end : end - i;
            emit(i, L, false, first);
            i += L;
        }
        else
        {
            // Plane-aligned: whole planes only, capped at the batch end
            // (the fiber operand re-bases there).
            Index const batch_end = (i / (n * k) + 1) * n * k;
            Index const cap = batch_end < end ? batch_end : end;
            Index planes = (cap - i) / k;
            if(planes >= 1)
            {
                emit(i, planes * k, true, first);
                i += planes * k;
            }
            else
            {
                // Trailing partial plane starting aligned: (m, 1, L, 1)
                // with fiber offset 0.
                Index const to_plane_end = (i / k + 1) * k - i;
                Index const L = to_plane_end < end - i ? to_plane_end
                                                       : end - i;
                emit(i, L, false, first);
                i += L;
            }
        }
    }
}

/*! Per-panel staging of one slice operand (softmax's maxsumexp input,
 * add/scale slice sources, reduction dst slices): the driver advances
 * panel by panel; a segment inside one chunk is used in place, a
 * straddling segment is gathered once into a bounded seg-element temp
 * (zero-initialized instead of gathered when init_zero - the beta==0 dst
 * case, whose pieces write every element but may legally skip writes on
 * empty pieces) and scattered back at panel change for writable stages.
 * The fiber class stages its k-element BATCH segments through the same
 * helper (one segment per batch, sub-ranges addressed by adding the
 * in-plane fiber offset at the call site).
 */
template<typename T, typename Arena>
struct SliceStage
{
    ChunkView<T> const *cv = nullptr;
    std::size_t seg = 0; //!< segment elements per panel
    bool writable = false;
    Arena *arena = nullptr;
    T *temp = nullptr; //!< lazily allocated on the first straddling panel
    Index active = -1; //!< panel currently staged (direct or temp)
    bool staged = false;

    SliceStage(ChunkView<T> const &c, std::size_t seg_elems, bool write,
        Arena &a):
        cv(&c),
        seg(seg_elems),
        writable(write),
        arena(&a)
    {
    }

    //! Pointer to panel's segment, staging it if it straddles a chunk.
    //! gather_old: pull the panel's current bytes into the temp first
    //! (read stages always gather - their temp feeds the kernel; a
    //! writable stage gathers only when the pieces accumulate onto the
    //! old bytes, and zero-initializes otherwise).
    T *advance(Index panel, bool gather_old)
    {
        if(panel == active)
        {
            return staged ? temp : ptr_direct(panel);
        }
        retire();
        active = panel;
        if(seg == 0 || in_one_chunk(*cv, offset(panel), seg))
        {
            staged = false;
            return ptr_direct(panel);
        }
        if(temp == nullptr)
        {
            temp = arena->alloc(seg);
        }
        if(!arena->ok_state())
        {
            staged = true;
            return temp;
        }
        if(!writable || gather_old)
        {
            gather_range(*cv, offset(panel), seg, temp, *arena);
        }
        else
        {
            arena->zero(temp, seg);
        }
        staged = true;
        return temp;
    }

    //! Scatter the active staged segment back (writable stages only).
    void retire() noexcept
    {
        if(staged && writable && active >= 0 && arena->ok_state())
        {
            scatter_range(*cv, offset(active), seg, temp, *arena);
        }
        staged = false;
        active = -1;
    }

    std::size_t offset(Index panel) const noexcept
    {
        return static_cast<std::size_t>(panel) * seg;
    }
    T *ptr_direct(Index panel) const
    {
        std::size_t const s = offset(panel);
        std::size_t const c = cv->chunk_of(s);
        return cv->ptr(c, s - cv->chunk_begin(c));
    }
};

} // namespace nntile::haul::chunkunits

#endif // NNTILE_USE_NNHAUL
