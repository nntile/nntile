/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/gemm.cc
 * GEMM operation for StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding headers
#include "nntile/nnhaul/ops/gemm.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Third-party headers

// Other NNTile headers
#include "nntile/kernel/cblas.hh"
#include "nntile/kernel/cublas.hh"

#include <algorithm>
#include <vector>

namespace nntile::haul
{

//! Constructor
template<typename T>
Gemm<std::tuple<T>>::Gemm():
    codelet(
        "nntile_gemm",
        &Gemm<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Gemm<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Gemm<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default.
    // Unsupported types no-op inside the CPU kernel.
}

// ---------------------------------------------------------------------------
// Native chunked GEMM driver
//
// Operand storage, derived from kernel::cblas::gemm / kernel::cublas::gemm
// (NOT assumed): the kernels implement the C-order (row-major) BLAS
// convention - cblas_*gemm(CblasRowMajor, opA, opB, m, n, k, A, ldA, B,
// ldB, C, ldC=n) and the mirrored cublasGemmStridedBatchedEx(transB,
// transA, n, m, k, B, A, C, ldC=n). Per batch element b the flat element
// offsets are therefore
//
//   A: NoTrans  A(i,x) = flatA[b*m*k + i*k + x]   (stored (m, k), ld=k)
//      Trans    A(i,x) = flatA[b*m*k + x*m + i]   (stored (k, m), ld=m)
//   B: NoTrans  B(x,j) = flatB[b*k*n + x*n + j]   (stored (k, n), ld=n)
//      Trans    B(x,j) = flatB[b*k*n + j*k + x]   (stored (n, k), ld=k)
//   C:          C(i,j) = flatC[b*m*n + i*n + j]   (stored (m, n), ld=n)
//
// with batch strides m*k / k*n / m*n (0 for a broadcast operand, whose
// panel pattern is then identical for every batch).
//
// The lds are deduced from (m, n, k, trans) by the kernels, so a sub-call
// gemm(m', n', k', pA, pB, pC) reads/writes PACKED sub-matrices: the only
// ld-valid sub-blocks of the layouts above are whole-row runs:
//
//   A NoTrans : rows [i0,i1) x all k columns  (m' = i1-i0, k' = k)
//   A Trans   : rows [x0,x1) x all m columns  (k' = x1-x0, m' = m)
//   B NoTrans : rows [x0,x1) x all n columns  (k' = x1-x0, n' = n)
//   B Trans   : rows [j0,j1) x all k columns  (n' = j1-j0, k' = k)
//   C         : rows [i0,i1) x all n columns  (m' = i1-i0, n' = n)
//
// Matching the (m', n', k') of all three operands per sub-call leaves
// exactly one splittable index per trans combination:
//
//   transA=NoTrans (any transB): split over i - C rows [i0,i1) pair with
//       A rows [i0,i1); k' and n' are forced full, so B must be one
//       contiguous run per batch (its whole matrix).
//   transA=Trans, transB=NoTrans: split over x - A rows [x0,x1) pair with
//       B rows [x0,x1); m' and n' are forced full, so C must be one
//       contiguous run (whole matrix), and the x-panels chain over k with
//       beta (first panel keeps the caller's beta, the rest use 1).
//   transA=Trans, transB=Trans: no free index; A, B and C must each be one
//       contiguous run per batch.
//
// "One contiguous run" for a chunked buffer means inside a single cache
// chunk; an operand whose whole matrix spans chunks (or a split operand's
// row that crosses a chunk boundary, including rows longer than a chunk)
// is staged through a body-local scratch: a whole-matrix gather for the
// forced-whole operands, a single-row gather/scatter for straddling rows
// of the split operands (bounded by the row length m/k/n, allocated once
// per body). All staging happens inside the one task body, so no extra
// tasks are submitted and the unchunked path stays byte-identical to the
// classic single call.
//
// The panel decomposition re-streams the whole operand once per extra
// panel (B for the i-split, C twice for every chained k-panel of the
// k-split) and copies every straddling row through the temp. When that
// traffic exceeds the two extra gathers the whole-operand treatment
// would cost (a pure function of the chunk segmentations and the call
// shape), the driver instead gathers the split operands too and issues
// one full-shape kernel call per batch - still inside the single body,
// still with the unchunked kernel shapes, so the results are bitwise
// the fast path's and the decision reproduces identically in a
// panel-forced contiguous run. Under the default ladder that fallback
// takes e.g. a 2048^3 NoTrans,NoTrans gemm (64 x 256 KiB chunks per
// operand: re-streaming B 64 times costs ~1 GiB against ~96 MiB of
// gathers), while few-panel shapes (a 2 MiB A/C around a 1 MiB B, like
// the chunked-tile graph tests) keep the pure panel path.
//
// Test hooks: NNHAUL_GEMM_FORCE_PANEL_BYTES=<n> virtually re-chunks every
// UNCHUNKED operand into n-byte panels (same arithmetic, one contiguous
// block), so a contiguous-mode process can reproduce the exact panel
// factorization of a chunked-mode run for bitwise A/B comparisons;
// NNHAUL_GEMM_FORCE_PANELS=1 pins the panel path on wherever the
// traffic model would gather, so the panel driver and its straddle
// staging can be exercised bitwise on any chunk class.
// ---------------------------------------------------------------------------

namespace
{

//! Panel-factorization test hook (see the block comment above).
std::size_t gemm_forced_chunk_bytes() noexcept
{
    char const *e = std::getenv("NNHAUL_GEMM_FORCE_PANEL_BYTES");
    if(e == nullptr)
    {
        return 0;
    }
    long const v = std::atol(e);
    return v > 0 ? static_cast<std::size_t>(v) : std::size_t{0};
}

//! Test hook: skip the adaptive whole-operand fallback so the panel
//! driver (and its straddle staging) runs even where the traffic model
//! below would gather through scratch.
bool gemm_force_panels() noexcept
{
    char const *e = std::getenv("NNHAUL_GEMM_FORCE_PANELS");
    return e != nullptr && std::atoi(e) != 0;
}

//! Chunk residency of one operand buffer's flat element space.
template<typename T>
struct ChunkView
{
    ::nnhaul::Buffer *buf = nullptr;
    std::size_t chunk_elems = 1; //!< elements of a full chunk
    std::size_t nchunks = 1;
    std::size_t total_elems = 0;
    bool real = false; //!< true: per-chunk pointers; false: one block

    ChunkView() = default;
    ChunkView(::nnhaul::Buffer *b, std::size_t force_chunk_elems):
        buf(b),
        total_elems(b->nbytes() / sizeof(T)),
        nchunks(b->chunk_count())
    {
        if(nchunks > 1)
        {
            chunk_elems = b->chunk_bytes(0) / sizeof(T);
            real = true;
        }
        else if(force_chunk_elems >= sizeof(T)
                && force_chunk_elems / sizeof(T) < total_elems)
        {
            // Test-only virtual factorization of an unchunked buffer.
            chunk_elems = force_chunk_elems / sizeof(T);
            nchunks = (total_elems + chunk_elems - 1) / chunk_elems;
            real = false;
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
        std::size_t e = (c + 1) * chunk_elems;
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
{
    if(len == 0)
    {
        return true;
    }
    std::size_t const c = cv.chunk_of(f);
    return f + len <= cv.chunk_end(c);
}

//! Invoke f(chunk, offset_in_chunk, offset_in_range, piece_len) for every
//! chunk-resident piece of the flat element range [start, start+len).
template<typename T, typename F>
void for_each_piece(ChunkView<T> const &cv, std::size_t start,
    std::size_t len, F &&f)
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
        f(c, at - cv.chunk_begin(c), done, n);
        done += n;
    }
}

//! One split operand's stored rows: count rows of len elements, row r at
//! flat element offset base + r*len (base is the batch matrix offset).
template<typename T>
struct RowView
{
    ChunkView<T> const *cv = nullptr;
    std::size_t base = 0;
    Index count = 0;
    std::size_t len = 0;

    //! True when row r crosses a chunk boundary (never fits one chunk).
    bool straddles(Index r) const noexcept
    {
        if(len == 0)
        {
            return false;
        }
        std::size_t const s = base + static_cast<std::size_t>(r) * len;
        return !in_one_chunk(*cv, s, len);
    }

    //! Whole rows of the chunk containing row r (>= 1; row r is clean).
    Index run(Index r) const noexcept
    {
        if(len == 0)
        {
            return count - r;
        }
        std::size_t const s = base + static_cast<std::size_t>(r) * len;
        std::size_t const c = cv->chunk_of(s);
        Index j = static_cast<Index>((cv->chunk_end(c) - s) / len);
        if(j > count - r)
        {
            j = count - r;
        }
        return j < 1 ? 1 : j;
    }

    //! Pointer to row r (valid only when !straddles(r)).
    T *ptr(Index r) const
    {
        std::size_t const s = base + static_cast<std::size_t>(r) * len;
        std::size_t const c = cv->chunk_of(s);
        return cv->ptr(c, s - cv->chunk_begin(c));
    }

    std::size_t row_offset(Index r) const noexcept
    {
        return base + static_cast<std::size_t>(r) * len;
    }
};

//! Walk the shared row index [0, rows) of two split operands: emit
//! (begin, len, p_straddle, q_straddle) panels whose rows lie in single
//! chunks of both operands; a row straddling in either operand becomes a
//! len-1 panel with the corresponding flag set (the caller stages that
//! row through the per-body temp). An empty index space emits one empty
//! panel so degenerate shapes (k=0 / m=0) still apply beta to C.
template<typename T, typename Emit>
void walk_row_panels(RowView<T> const &p, RowView<T> const &q, Index rows,
    Emit &&emit)
{
    if(rows <= 0)
    {
        emit(Index{0}, Index{0}, false, false);
        return;
    }
    Index i = 0;
    while(i < rows)
    {
        bool const ps = p.straddles(i);
        bool const qs = q.straddles(i);
        if(ps || qs)
        {
            emit(i, Index{1}, ps, qs);
            ++i;
            continue;
        }
        Index len = std::min(p.run(i), q.run(i));
        emit(i, len, false, false);
        i += len;
    }
}

//! Panel census of the same walk (for the traffic model below): returns
//! the panel count and the elements staged through the straddle temps.
template<typename T>
void count_row_panels(RowView<T> const &p, RowView<T> const &q, Index rows,
    Index &panels, std::size_t &straddle_elems)
{
    panels = 0;
    straddle_elems = 0;
    walk_row_panels(p, q, rows,
        [&](Index, Index len, bool ps, bool qs)
        {
            ++panels;
            if(ps)
            {
                straddle_elems += p.len;
            }
            if(qs)
            {
                straddle_elems += q.len;
            }
            (void)len;
        });
}

//! Host-memory scratch arena for the CPU body.
template<typename T>
struct ArenaCpu
{
    std::vector<std::vector<T>> blocks;

    T *alloc(std::size_t elems)
    {
        blocks.emplace_back(elems == 0 ? 1 : elems);
        return blocks.back().data();
    }
    void copy(T *dst, T const *src, std::size_t elems) noexcept
    {
        if(elems > 0)
        {
            std::memcpy(dst, src, elems * sizeof(T));
        }
    }
    bool ok_state() const noexcept
    {
        return true;
    }
};

#ifdef NNTILE_USE_CUDA
//! Stream-ordered device scratch arena for the CUDA body: every block is
//! cudaMallocAsync'd on the compute stream and freed (cudaFreeAsync) at
//! scope end, so all gather/kernel/scatter work enqueued in between is
//! stream-ordered before the release.
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
            // Same convention as the other wrapper error paths (e.g.
            // torch_dispatch bodies): report on stderr; the codelet
            // returns void, so a failed scratch leaves the result
            // unwritten (the ok flag stops the driver's further calls).
            std::fprintf(stderr,
                "nntile_gemm: cudaMallocAsync failed: %s\n",
                cudaGetErrorString(err));
            ok = false;
            return nullptr;
        }
        blocks.push_back(p);
        return p;
    }
    void copy(T *dst, T const *src, std::size_t elems) noexcept
    {
        if(elems == 0 || !ok)
        {
            return;
        }
        cudaError_t const err = cudaMemcpyAsync(dst, src, elems * sizeof(T),
            cudaMemcpyDeviceToDevice, stream);
        if(err != cudaSuccess)
        {
            std::fprintf(stderr,
                "nntile_gemm: cudaMemcpyAsync failed: %s\n",
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
                    "nntile_gemm: cudaFreeAsync failed: %s\n",
                    cudaGetErrorString(err));
            }
        }
    }
};
#endif // NNTILE_USE_CUDA

//! Gather the flat element range [start, start+len) of a chunked buffer
//! into one contiguous scratch row.
template<typename T, typename Arena>
void gather_row(ChunkView<T> const &cv, std::size_t start, std::size_t len,
    T *dst, Arena &arena) noexcept
{
    for_each_piece(cv, start, len,
        [&](std::size_t c, std::size_t off, std::size_t dst_off,
            std::size_t n) { arena.copy(dst + dst_off, cv.ptr(c, off), n); });
}

//! Mirror of gather_row: scatter a staged row back to its chunks.
template<typename T, typename Arena>
void scatter_row(ChunkView<T> const &cv, std::size_t start, std::size_t len,
    T const *src, Arena &arena) noexcept
{
    for_each_piece(cv, start, len,
        [&](std::size_t c, std::size_t off, std::size_t src_off,
            std::size_t n) { arena.copy(cv.ptr(c, off), src + src_off, n); });
}

//! A forced-whole operand: every batch matrix must be one contiguous run;
//! matrices that span chunks all go through a whole-buffer scratch
//! (gathered up front for inputs and for C when beta != 0, scattered back
//! afterwards when the operand is written).
template<typename T>
struct WholeView
{
    ChunkView<T> const *cv = nullptr;
    std::size_t stride = 0; //!< per-batch matrix elements
    bool broadcast = false;
    T *scratch = nullptr; //!< whole-buffer scratch, null when native

    template<typename Arena>
    void init(Arena &arena, Index batch, bool gather_needed)
    {
        for(Index b = 0; b < batch; ++b)
        {
            std::size_t const base =
                broadcast ? std::size_t{0} : static_cast<std::size_t>(b) * stride;
            if(!in_one_chunk(*cv, base, stride))
            {
                scratch = arena.alloc(cv->total_elems);
                break;
            }
        }
        if(scratch != nullptr && gather_needed)
        {
            for(std::size_t c = 0; c < cv->nchunks; ++c)
            {
                std::size_t const len = cv->chunk_end(c) - cv->chunk_begin(c);
                if(len > 0)
                {
                    arena.copy(scratch + cv->chunk_begin(c), cv->ptr(c, 0), len);
                }
            }
        }
    }

    //! Pointer to batch b's whole matrix (scratch or chunk-resident).
    T *ptr(Index b) const
    {
        std::size_t const base =
            broadcast ? std::size_t{0} : static_cast<std::size_t>(b) * stride;
        if(scratch != nullptr)
        {
            return scratch + base;
        }
        std::size_t const c = cv->chunk_of(base);
        return cv->ptr(c, base - cv->chunk_begin(c));
    }

    template<typename Arena>
    void scatter_back(Arena &arena) const noexcept
    {
        if(scratch == nullptr)
        {
            return;
        }
        for(std::size_t c = 0; c < cv->nchunks; ++c)
        {
            std::size_t const len = cv->chunk_end(c) - cv->chunk_begin(c);
            if(len > 0)
            {
                arena.copy(cv->ptr(c, 0), scratch + cv->chunk_begin(c), len);
            }
        }
    }
};

//! The native chunked driver (see the layout derivation at the top).
//! sub(m', n', k', pA, pB, pC, beta_eff) must issue the existing kernel
//! with batch=1 on those packed panel pointers.
template<typename T, typename Arena, typename Sub>
void gemm_chunked(TransOp transA, TransOp transB, Index m, Index n, Index k,
    Index batch, Scalar alpha, Scalar beta, bool broadcast_a,
    bool broadcast_b, ::nnhaul::Buffer *bufA, ::nnhaul::Buffer *bufB,
    ::nnhaul::Buffer *bufC, std::size_t force_chunk_elems, Arena &arena,
    Sub &&sub)
{
    bool const tA = transA.value == TransOp::Trans;
    bool const tB = transB.value == TransOp::Trans;
    (void)alpha; // applied by the caller's sub-call lambda
    ChunkView<T> const cvA(bufA, force_chunk_elems);
    ChunkView<T> const cvB(bufB, force_chunk_elems);
    ChunkView<T> const cvC(bufC, force_chunk_elems);
    std::size_t const stride_a = static_cast<std::size_t>(m) * k;
    std::size_t const stride_b = static_cast<std::size_t>(k) * n;
    std::size_t const stride_c = static_cast<std::size_t>(m) * n;
    // Whole-operand execution (the transA=Trans,transB=Trans shape, and
    // the adaptive fallback of the splittable shapes): every batch
    // matrix as one contiguous run, gathering through the arena scratch
    // those operands whose matrices span chunks, one kernel call per
    // batch. Still zero extra tasks and the kernel calls match the
    // unchunked shapes, so the results are bitwise the fast path's.
    auto run_whole = [&]()
    {
        WholeView<T> whole_a;
        whole_a.cv = &cvA;
        whole_a.stride = stride_a;
        whole_a.broadcast = broadcast_a;
        whole_a.init(arena, batch, true);
        WholeView<T> whole_b;
        whole_b.cv = &cvB;
        whole_b.stride = stride_b;
        whole_b.broadcast = broadcast_b;
        whole_b.init(arena, batch, true);
        WholeView<T> whole_c;
        whole_c.cv = &cvC;
        whole_c.stride = stride_c;
        whole_c.init(arena, batch, beta != Scalar{0});
        if(!arena.ok_state())
        {
            return;
        }
        for(Index b = 0; b < batch; ++b)
        {
            sub(m, n, k, whole_a.ptr(b), whole_b.ptr(b), whole_c.ptr(b), beta);
        }
        whole_c.scatter_back(arena);
    };
    if(tA && tB)
    {
        // ---- transA=Trans, transB=Trans: no free index, every operand's
        // whole matrix must be one contiguous run -----------------------
        run_whole();
        return;
    }
    if(!gemm_force_panels())
    {
        // Traffic model: the panel driver re-streams the whole operand
        // once per extra panel (B here; C read+write per chained k-panel
        // for the k-split) and stages every straddling row through the
        // temp, while whole-operand execution additionally gathers and
        // (for C) scatters the two split operands. Pick the cheaper side
        // of that trade; the model is a pure function of the chunk
        // segmentations and the call shape, so the panel-forced
        // contiguous A/B runs (NNHAUL_GEMM_FORCE_PANEL_BYTES) decide
        // identically and stay bitwise comparable.
        Index panels = 0;
        std::size_t straddle_elems = 0;
        for(Index b = 0; b < batch; ++b)
        {
            Index p_b = 0;
            std::size_t s_b = 0;
            if(tA)
            {
                RowView<T> const rows_a{&cvA,
                    broadcast_a ? std::size_t{0}
                                : static_cast<std::size_t>(b) * stride_a,
                    k, static_cast<std::size_t>(m)};
                RowView<T> const rows_b{&cvB,
                    broadcast_b ? std::size_t{0}
                                : static_cast<std::size_t>(b) * stride_b,
                    k, static_cast<std::size_t>(n)};
                count_row_panels(rows_a, rows_b, k, p_b, s_b);
            }
            else
            {
                RowView<T> const rows_a{&cvA,
                    broadcast_a ? std::size_t{0}
                                : static_cast<std::size_t>(b) * stride_a,
                    m, static_cast<std::size_t>(k)};
                RowView<T> const rows_c{&cvC,
                    static_cast<std::size_t>(b) * stride_c, m,
                    static_cast<std::size_t>(n)};
                count_row_panels(rows_a, rows_c, m, p_b, s_b);
            }
            panels += p_b;
            straddle_elems += s_b;
        }
        std::size_t const bytes_a = cvA.total_elems * sizeof(T);
        std::size_t const bytes_b = cvB.total_elems * sizeof(T);
        std::size_t const bytes_c = cvC.total_elems * sizeof(T);
        std::size_t const straddle_bytes = 2 * straddle_elems * sizeof(T);
        std::size_t const extra_panels =
            panels > 1 ? static_cast<std::size_t>(panels - 1) : 0;
        bool whole;
        if(tA)
        {
            // k-split: whole adds gathers of A and B, panels re-stream C
            // (read+write) once per extra k-panel.
            whole = 2 * (bytes_a + bytes_b)
                < 2 * extra_panels * bytes_c + straddle_bytes;
        }
        else
        {
            // i-split: whole adds gathers of A and C, panels re-stream B
            // once per extra i-panel.
            whole = 2 * (bytes_a + bytes_c)
                < extra_panels * bytes_b + straddle_bytes;
        }
        if(whole)
        {
            run_whole();
            return;
        }
    }
    if(!tA)
    {
        // ---- split over i: A rows [i0,i1) x full k, C rows [i0,i1) x
        // full n, B's whole matrix as one contiguous run ---------------
        WholeView<T> whole_b;
        whole_b.cv = &cvB;
        whole_b.stride = stride_b;
        whole_b.broadcast = broadcast_b;
        whole_b.init(arena, batch, true);
        T *temp_a =
            cvA.nchunks > 1 ? arena.alloc(static_cast<std::size_t>(k)) : nullptr;
        T *temp_c =
            cvC.nchunks > 1 ? arena.alloc(static_cast<std::size_t>(n)) : nullptr;
        if(!arena.ok_state())
        {
            return;
        }
        for(Index b = 0; b < batch; ++b)
        {
            RowView<T> rows_a{&cvA,
                broadcast_a ? std::size_t{0}
                            : static_cast<std::size_t>(b) * stride_a,
                m, static_cast<std::size_t>(k)};
            RowView<T> rows_c{&cvC,
                static_cast<std::size_t>(b) * stride_c, m,
                static_cast<std::size_t>(n)};
            T *pb = whole_b.ptr(b);
            walk_row_panels(rows_a, rows_c, m,
                [&](Index i0, Index len, bool a_str, bool c_str) {
                    if(!arena.ok_state())
                    {
                        return;
                    }
                    T const *pa = nullptr;
                    if(a_str)
                    {
                        gather_row(cvA, rows_a.row_offset(i0),
                            static_cast<std::size_t>(k), temp_a, arena);
                        pa = temp_a;
                    }
                    else
                    {
                        pa = rows_a.ptr(i0);
                    }
                    T *pc = nullptr;
                    if(c_str)
                    {
                        if(beta != Scalar{0})
                        {
                            gather_row(cvC, rows_c.row_offset(i0),
                                static_cast<std::size_t>(n), temp_c, arena);
                        }
                        pc = temp_c;
                    }
                    else
                    {
                        pc = rows_c.ptr(i0);
                    }
                    sub(len, n, k, pa, pb, pc, beta);
                    if(c_str)
                    {
                        scatter_row(cvC, rows_c.row_offset(i0),
                            static_cast<std::size_t>(n), temp_c, arena);
                    }
                });
        }
        return;
    }
    if(!tB)
    {
        // ---- split over x: A rows [x0,x1) x full m (stored (k,m)), B
        // rows [x0,x1) x full n (stored (k,n)), C's whole matrix as one
        // contiguous run; the x-panels chain over k with beta ----------
        WholeView<T> whole_c;
        whole_c.cv = &cvC;
        whole_c.stride = stride_c;
        whole_c.init(arena, batch, beta != Scalar{0});
        T *temp_a =
            cvA.nchunks > 1 ? arena.alloc(static_cast<std::size_t>(m)) : nullptr;
        T *temp_b =
            cvB.nchunks > 1 ? arena.alloc(static_cast<std::size_t>(n)) : nullptr;
        if(!arena.ok_state())
        {
            return;
        }
        for(Index b = 0; b < batch; ++b)
        {
            RowView<T> rows_a{&cvA,
                broadcast_a ? std::size_t{0}
                            : static_cast<std::size_t>(b) * stride_a,
                k, static_cast<std::size_t>(m)};
            RowView<T> rows_b{&cvB,
                broadcast_b ? std::size_t{0}
                            : static_cast<std::size_t>(b) * stride_b,
                k, static_cast<std::size_t>(n)};
            T *pc = whole_c.ptr(b);
            bool first = true;
            walk_row_panels(rows_a, rows_b, k,
                [&](Index x0, Index len, bool a_str, bool b_str) {
                    if(!arena.ok_state())
                    {
                        return;
                    }
                    T const *pa = nullptr;
                    if(a_str)
                    {
                        gather_row(cvA, rows_a.row_offset(x0),
                            static_cast<std::size_t>(m), temp_a, arena);
                        pa = temp_a;
                    }
                    else
                    {
                        pa = rows_a.ptr(x0);
                    }
                    T const *pb = nullptr;
                    if(b_str)
                    {
                        gather_row(cvB, rows_b.row_offset(x0),
                            static_cast<std::size_t>(n), temp_b, arena);
                        pb = temp_b;
                    }
                    else
                    {
                        pb = rows_b.ptr(x0);
                    }
                    sub(m, n, len, pa, pb, pc, first ? beta : Scalar{1});
                    first = false;
                });
        }
        whole_c.scatter_back(arena);
        return;
    }
    // Unreachable: tA&&tB returned at run_whole above, !tA and tA&&!tB
    // returned in their split branches.
}

} // namespace

#ifdef NNTILE_USE_CBLAS // CPU implementation requires CBLAS
//! GEMM for contiguous matrices without padding through StarPU buffers
template<typename T>
void Gemm<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_a = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_b = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_c = static_cast<::nnhaul::Buffer *>(buffers[2]);
    // Launch kernel
    if constexpr (kernel::cblas::gemm_is_supported<T>)
    {
        std::size_t const forced = gemm_forced_chunk_bytes();
        if(buf_a->chunk_count() == 1 && buf_b->chunk_count() == 1
            && buf_c->chunk_count() == 1 && forced == 0)
        {
            const T *A = buf_a->get_ptr<T>();
            const T *B = buf_b->get_ptr<T>();
            T *C = buf_c->get_ptr<T>();
            kernel::cblas::gemm<T>(
                args->transA,
                args->transB,
                args->m,
                args->n,
                args->k,
                args->batch,
                args->alpha,
                A,
                B,
                args->beta,
                C,
                args->broadcast_a,
                args->broadcast_b
            );
            return;
        }
        try
        {
            ArenaCpu<T> arena;
            gemm_chunked<T>(
                args->transA,
                args->transB,
                args->m,
                args->n,
                args->k,
                args->batch,
                args->alpha,
                args->beta,
                args->broadcast_a,
                args->broadcast_b,
                buf_a,
                buf_b,
                buf_c,
                forced,
                arena,
                [&](Index mm, Index nn, Index kk, T const *pa, T const *pb,
                    T *pc, Scalar beta_eff) {
                    kernel::cblas::gemm<T>(
                        args->transA,
                        args->transB,
                        mm,
                        nn,
                        kk,
                        Index{1},
                        args->alpha,
                        pa,
                        pb,
                        beta_eff,
                        pc
                    );
                });
            note_gemm_native_chunked();
        }
        catch(...)
        {
            // A noexcept body must not throw; a failed scratch allocation
            // leaves the result unwritten (same failure class as a failed
            // kernel launch).
            std::fprintf(stderr,
                    "nntile_gemm: cpu chunked driver aborted after an "
                    "allocation failure; C left unwritten\n");
        }
    }
}
#endif // NNTILE_USE_CBLAS

#ifdef NNTILE_USE_CUDA // CUDA implementation requires cuBLAS
//! GEMM for contiguous matrices without padding through StarPU buffers
template<typename T>
void Gemm<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_a = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_b = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_c = static_cast<::nnhaul::Buffer *>(buffers[2]);
    // Get cuBLAS handle and CUDA stream
    cublasHandle_t handle = ::nnhaul::cublas_handle();
    cudaStream_t stream = ::nnhaul::cuda_stream();
    cublasSetStream(handle, stream);
    // alpha and beta parameters of GEMM operation are on CPU host
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    std::size_t const forced = gemm_forced_chunk_bytes();
    if(buf_a->chunk_count() == 1 && buf_b->chunk_count() == 1
        && buf_c->chunk_count() == 1 && forced == 0)
    {
        const T *A = buf_a->get_ptr<T>();
        const T *B = buf_b->get_ptr<T>();
        T *C = buf_c->get_ptr<T>();
        // Call corresponding cuBLAS routine
        kernel::cublas::gemm<T>(
            handle,
            args->transA,
            args->transB,
            args->m,
            args->n,
            args->k,
            args->batch,
            args->alpha,
            A,
            B,
            args->beta,
            C,
            args->broadcast_a,
            args->broadcast_b
        );
        return;
    }
    ArenaCuda<T> arena(stream);
    gemm_chunked<T>(
        args->transA,
        args->transB,
        args->m,
        args->n,
        args->k,
        args->batch,
        args->alpha,
        args->beta,
        args->broadcast_a,
        args->broadcast_b,
        buf_a,
        buf_b,
        buf_c,
        forced,
        arena,
        [&](Index mm, Index nn, Index kk, T const *pa, T const *pb, T *pc,
            Scalar beta_eff) {
            kernel::cublas::gemm<T>(
                handle,
                args->transA,
                args->transB,
                mm,
                nn,
                kk,
                Index{1},
                args->alpha,
                pa,
                pb,
                beta_eff,
                pc
            );
        });
    if(arena.ok)
    {
        note_gemm_native_chunked();
    }
    else
    {
        std::fprintf(stderr,
            "nntile_gemm: chunked CUDA driver aborted after a stream "
            "error; C left unwritten\n");
    }
}
#endif //NNTILE_USE_CUDA

//! Footprint for GEMM tasks that depends on transA, transB, M, N, K, batch and alpha
template<typename T>
std::uint64_t Gemm<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // In case alpha is zero, entire gemm is unnecessary so it is better to
    // give it a different footprint since gemm time will be totally different
    uint32_t hash = args->alpha == Scalar{0} ? -1 : 0;
    // Single codelet is used for all combinations of transA and transB
    hash = ::nntile::haul::fnv1a(&args->transA, sizeof(args->transA), hash);
    hash = ::nntile::haul::fnv1a(&args->transB, sizeof(args->transB), hash);
    // Apply hash over parameters M, N and K. This way if we swap values of M,
    // N and K total size of buffers will remain the same, but the footprint
    // will be different
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    hash = ::nntile::haul::fnv1a(&args->batch, sizeof(args->batch), hash);
    return hash;
}

template<typename T>
void Gemm<std::tuple<T>>::submit(int starpu_worker_hint, const TransOp &transA, const TransOp &transB, Index m, Index n,
        Index k, Index batch, Scalar alpha, ::nnhaul::Handle & A, ::nnhaul::Handle & B, Scalar beta,
        ::nnhaul::Handle & C, int redux, bool broadcast_a, bool broadcast_b)
{
    if(redux != 0)
    {
        throw std::runtime_error("redux != 0 is not supported");
    }
    // Check that matrix sizes fit proper types for underlying CBLAS
#ifdef NNTILE_USE_CBLAS
    if(static_cast<CBLAS_INT>(m) != m)
    {
        throw std::runtime_error("GEMM size M does not fit CBLAS_INT");
    }
    if(static_cast<CBLAS_INT>(n) != n)
    {
        throw std::runtime_error("GEMM size N does not fit CBLAS_INT");
    }
    if(static_cast<CBLAS_INT>(k) != k)
    {
        throw std::runtime_error("GEMM size K does not fit CBLAS_INT");
    }
#endif // NNTILE_USE_CBLAS
    constexpr Scalar zero = 0, one = 1;
    enum starpu_data_access_mode C_mode;
    if(beta == zero)
    {
        C_mode = STARPU_W;
    }
    else if(beta == one)
    {
        if(redux != 0)
        {
            C_mode = STARPU_REDUX;
        }
        else
        {
            C_mode = static_cast<starpu_data_access_mode>(
                STARPU_RW | STARPU_COMMUTE);
        }
    }
    else
    {
        C_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->transA = transA;
    args->transB = transB;
    args->m = m;
    args->n = n;
    args->k = k;
    args->batch = batch;
    args->alpha = alpha;
    args->beta = beta;
    args->broadcast_a = broadcast_a;
    args->broadcast_b = broadcast_b;
    // FLOPs calculation
    double nflops = 2 * m * n * k * batch;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked panel driver; gemm never goes through the staging fallback.
    (void)nflops;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &A }, { STARPU_R, &B }, { C_mode, &C } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Gemm<std::tuple<nntile::fp64_t>>;
template class Gemm<std::tuple<nntile::fp32_t>>;
template class Gemm<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Gemm<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Gemm<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Gemm<std::tuple<nntile::bf16_t>>;
template class Gemm<std::tuple<nntile::fp16_t>>;

//! Pack of gemm operations for different types
gemm_pack_t gemm;

} // namespace nntile::haul
