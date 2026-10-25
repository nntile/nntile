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
 * @version 1.1.0
 * */

#pragma once

#ifdef NNTILE_USE_NNHAUL

#include <nntile/base_types.hh>
#include <nntile/defs.h>

#include <nnhaul/nnhaul.hh>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#ifdef NNTILE_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nntile::haul::chunkunits
{

//! Test hook (the gemm FORCE_PANEL_BYTES scheme): when positive, every
//! UNCHUNKED buffer is virtually re-chunked into chunks of that many
//! bytes, so a contiguous-mode process reproduces the exact unit
//! factorization of a chunked-mode run for bitwise A/B comparisons.
//! Read per call like the gemm knob.
inline std::size_t forced_chunk_bytes() noexcept
{
    char const *e = std::getenv("NNHAUL_CHUNK_UNITS_FORCE_BYTES");
    if(e == nullptr)
    {
        return 0;
    }
    long const v = std::atol(e);
    return v > 0 ? static_cast<std::size_t>(v) : std::size_t{0};
}

//! The nnhaul chunk-kernel element type matching the nntile tensor type.
//! The accelerated fp32 family stores a plain float (repr_t float), so a
//! 4-byte element maps to F32 and an 8-byte one to F64; the 16-bit types
//! are distinguished by name (bf16/fp16 have the same storage size).
template<typename T>
constexpr ::nnhaul::ElemType nnhaul_elem_type() noexcept
{
    using U = std::remove_cv_t<T>;
    if constexpr(std::is_same_v<U, bf16_t>)
    {
        return ::nnhaul::ElemType::BF16;
    }
    else if constexpr(std::is_same_v<U, fp16_t>)
    {
        return ::nnhaul::ElemType::F16;
    }
    else if constexpr(sizeof(U) == 8)
    {
        return ::nnhaul::ElemType::F64;
    }
    else if constexpr(sizeof(U) == 4)
    {
        return ::nnhaul::ElemType::F32;
    }
    else
    {
        static_assert(!std::is_same_v<U, U>,
            "nnhaul chunk kernels: unsupported element type");
        return ::nnhaul::ElemType::F32;
    }
}

//! floor(log2(v)) for v >= 1.
inline unsigned log2_floor(std::size_t v) noexcept
{
    unsigned s = 0;
    while(v > 1)
    {
        v >>= 1;
        ++s;
    }
    return s;
}

//! Chunk residency of one buffer's flat element space: equal-size chunks
//! with a shorter tail; an unchunked buffer is the 1-chunk degenerate case
//! (same arithmetic as the anonymous ChunkView of src/nnhaul/gemm.cc,
//! published here so the unit-structured families share one walker).
//!
//! Every production chunk class is kChunkClassBytes and a power of two,
//! and the element types are powers of two, so chunk_elems is normally a
//! power of two: chunk_of/chunk_begin then lower to a shift/mask instead
//! of a 64-bit division/multiply on the address path. The test hook
//! NNHAUL_CHUNK_UNITS_FORCE_BYTES may force an arbitrary (non-power-of-
//! two) size, so the exact division/multiply fallback is kept and is
//! selected once, at construction, by the `pow2` flag.
template<typename T>
struct ChunkView
{
    ::nnhaul::Buffer *buf = nullptr;
    std::size_t chunk_elems = 1; //!< elements of a full chunk
    std::size_t nchunks = 1;
    std::size_t total_elems = 0;
    bool real = false; //!< true: per-chunk pointers; false: one block
    bool pow2 = false; //!< chunk_elems is a power of two
    unsigned shift = 0; //!< log2(chunk_elems) when pow2
    std::size_t mask = 0; //!< chunk_elems - 1 when pow2

    ChunkView() = default;
    ChunkView(::nnhaul::Buffer *b, std::size_t force_chunk_bytes):
        buf(b),
        total_elems(b->nbytes() / sizeof(T)),
        nchunks(b->chunk_count())
    {
        if(nchunks > 1)
        {
            chunk_elems = b->chunk_bytes(0) / sizeof(T);
            real = true;
        }
        else if(force_chunk_bytes >= sizeof(T)
            && force_chunk_bytes / sizeof(T) < total_elems)
        {
            // Test-only virtual factorization of an unchunked buffer.
            chunk_elems = force_chunk_bytes / sizeof(T);
            nchunks = (total_elems + chunk_elems - 1) / chunk_elems;
            real = false;
        }
        else
        {
            chunk_elems = total_elems > 0 ? total_elems : 1;
            nchunks = 1;
            real = false;
        }
        if(chunk_elems != 0 && (chunk_elems & (chunk_elems - 1)) == 0)
        {
            pow2 = true;
            shift = log2_floor(chunk_elems);
            mask = chunk_elems - 1;
        }
    }

    std::size_t chunk_of(std::size_t elem) const noexcept
    {
        return pow2 ? elem >> shift : elem / chunk_elems;
    }
    std::size_t chunk_begin(std::size_t c) const noexcept
    {
        return pow2 ? c << shift : c * chunk_elems;
    }
    //! Offset of `elem` inside its chunk (chunk_of + chunk_begin folded).
    std::size_t chunk_off(std::size_t elem) const noexcept
    {
        return pow2 ? elem & mask : elem - (elem / chunk_elems) * chunk_elems;
    }
    std::size_t chunk_end(std::size_t c) const noexcept
    {
        std::size_t const e = chunk_begin(c + 1);
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
        arena.copy(dst + done, cv.ptr(c, cv.chunk_off(at)), n);
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
        arena.copy(cv.ptr(c, cv.chunk_off(at)), src + done, n);
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
        return cv->ptr(c, cv->chunk_off(s));
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
        return cv->ptr(c, cv->chunk_off(s));
    }
};

} // namespace nntile::haul::chunkunits

#endif // NNTILE_USE_NNHAUL
