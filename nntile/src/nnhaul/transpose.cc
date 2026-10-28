/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/transpose.cc
 * Transpose operation for StarPU buffers
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/transpose.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/transpose.hh"
#include "nntile/nnhaul/chunk_units.hh"
#include "nntile/nnhaul/stage.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked transpose driver (chunk_units.hh scheme + the panel
// kernel variant of the kernel layer).
//
// Kernel index map (derived from kernel::transpose::cpu/cuda, NOT
// assumed): src is m-by-n with element (i, j) at flat i + j*m (the i
// axis contiguous, column stride m) and dst is n-by-m with element
// (i, j) at flat i*n + j (the j axis contiguous, row stride n) - each
// side's contiguous unit is the OTHER side's strided index, so NO
// sub-block of the existing whole-matrix signature can express a
// piece; the panel kernel (kernel::transpose::panel) transposes an
// nr-by-nc sub-block given its (0, 0) corner base pointers while
// keeping the whole matrix's strides (src column stride m, dst row
// stride n). The driver walks the src by COLUMN units (m elements,
// UnitSpan rows = columns): runs of whole columns inside single src
// chunks become panel pieces; a column straddling a src chunk
// boundary is gathered into a bounded m temp and drives its own
// pieces. The dst side is refined per piece: for the column window
// [j0, j0+P) the dst flat window of row i is [i*n+j0, i*n+j0+P);
// consecutive rows are batched while the covered flat range stays
// inside one dst chunk (whole-chunk row runs), and the (at most two,
// since P never exceeds the equal-class chunk size) sub-windows of a
// straddling row are split at the boundary. Every element moves
// exactly once, elementwise and deterministic, so the piecewise
// result is BITWISE the whole-buffer call under any factorization.
// Zero staging; all-unchunked buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void transpose_chunked(Index m, Index n, Scalar alpha,
    ::nnhaul::Buffer *buf_src, ::nnhaul::Buffer *buf_dst,
    Arena &arena, Sub &&sub)
{
    if(m <= 0 || n <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    chunkunits::UnitSpan<T> const span_src{&cv_src, std::size_t(m), Index{1}};
    bool const chunked_src = cv_src.nchunks > 1;
    T *temp_col = chunked_src ? arena.alloc(std::size_t(m)) : nullptr;
    // Drive the rows [i, i+R) of one src column window (P columns fully
    // inside one src chunk, base ps at row 0 of column j0) through
    // dst-chunk-resident panels. The src base of a piece is element
    // (i, j0) - ps + i stays inside the src chunk because the run only
    // covers whole columns.
    auto emit_rows = [&](T const *ps, Index j0, Index P)
    {
        Index i = 0;
        while(i < m && arena.ok_state())
        {
            std::size_t const f = std::size_t(i) * n + j0;
            std::size_t const c = cv_dst.chunk_of(f);
            std::size_t const end = cv_dst.chunk_end(c);
            T const *ps_i = ps + i;
            if(f + std::size_t(P) > end)
            {
                // Window straddles the boundary at end: split columns.
                Index const jb = static_cast<Index>(end - f);
                sub(m, n, Index{1}, jb, alpha, ps_i,
                    cv_dst.ptr(c, f - cv_dst.chunk_begin(c)));
                if(jb < P && arena.ok_state())
                {
                    std::size_t const f2 = f + jb;
                    std::size_t const c2 = cv_dst.chunk_of(f2);
                    // The trailing window starts jb columns further in,
                    // so its src base shifts by jb columns (jb*m src
                    // elements) too.
                    sub(m, n, Index{1}, P - jb, alpha, ps_i + jb * m,
                        cv_dst.ptr(c2, f2 - cv_dst.chunk_begin(c2)));
                }
                ++i;
                continue;
            }
            // Batch rows while the covered flat range stays in chunk c.
            std::size_t const last = std::size_t(m - 1) * n + j0 + P;
            std::size_t const capped_end = end < last ? end : last;
            Index R = static_cast<Index>(
                (capped_end - std::size_t(j0) - std::size_t(P)) / n) - i + 1;
            if(R < 1)
            {
                R = 1;
            }
            sub(m, n, R, P, alpha, ps_i,
                cv_dst.ptr(c, f - cv_dst.chunk_begin(c)));
            i += R;
        }
    };
    Index j = 0;
    while(j < n && arena.ok_state())
    {
        if(span_src.straddles(j))
        {
            chunkunits::gather_range(cv_src, std::size_t(m) * j,
                std::size_t(m), temp_col, arena);
            if(arena.ok_state())
            {
                emit_rows(temp_col, j, Index{1});
            }
            ++j;
            continue;
        }
        Index P = span_src.run(j);
        if(P > n - j)
        {
            P = n - j;
        }
        emit_rows(span_src.ptr(j), j, P);
        j += P;
    }
}

} // namespace

//! Constructor
template<typename T>
Transpose<std::tuple<T>>::Transpose():
    codelet(
        "nntile_transpose",
        &Transpose<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Transpose<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Transpose<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::transpose::cpu<T>
template<typename T>
void Transpose<std::tuple<T>>::cpu(void *buffers[], void *cl_args) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    if((buf_src->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_dst->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
       )
    {
        // Classic contiguous dispatch: illegal under the force flag
        // (the tripwire proves it never happens in a flag-on run).
        chunkunits::classic_dispatch_tripwire("nntile_transpose", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::transpose::cpu<T>(args->m, args->n, args->alpha,
            buf_src->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        transpose_chunked<T>(
            args->m,
            args->n,
            args->alpha,
            buf_src,
            buf_dst,
            arena,
            [](Index mm, Index nn, Index nr, Index nc, Scalar al,
                T const *ps, T *pd) {
                kernel::transpose::panel<T>(mm, nn, nr, nc, al, ps, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_transpose: cpu chunked driver aborted after an "
            "allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void Transpose<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Transpose<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Transpose<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::transpose::cuda<T>
template<typename T>
void Transpose<std::tuple<T>>::cuda(void *buffers[], void *cl_args) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if((buf_src->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_dst->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
       )
    {
        // Classic contiguous dispatch: illegal under the force flag
        // (the tripwire proves it never happens in a flag-on run).
        chunkunits::classic_dispatch_tripwire("nntile_transpose", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::transpose::cuda<T>(stream, args->m, args->n, args->alpha,
            buf_src->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    // Chunked single-launch path: the WHOLE transpose is issued as ONE
    // chunk-aware kernel with the chunk base-pointer tables read from the
    // shared device chunk-ptr ring (no chunk-count cap), instead of the
    // per-panel driver's O(src windows x dst row runs) launches. Elementwise
    // and bitwise the whole-buffer call. A launch whose pointer set exceeds
    // the ring, and the arbitrary factorizations,
    // keep the per-panel driver below as the fallback.
    // Piece C (2026-10-18c): the all-1-chunk degenerate launch is
    // the FIRST thing the chunked block does -- the chunk counts are
    // read straight off the buffers, before any
    // ChunkView/UnitSpan/RingTables work exists, so the degenerate
    // host path is as lean as the classic dispatch above (gemm
    // path-F precedent; launch pointer-identical, bitwise by
    // construction, ring untouched).
    if(buf_src->chunk_count() == 1 && buf_dst->chunk_count() == 1)
    {
        kernel::transpose::cuda<T>(stream, args->m, args->n, args->alpha,
            buf_src->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    {
        chunkunits::RingTables &tabs = chunkunits::ring_tables();
        tabs.clear();
        std::size_t const o_src_ptrs = tabs.add(cv_src);
        std::size_t const o_dst_ptrs = tabs.add(cv_dst);
        chunkunits::ChunkPtrRing &arena = chunkunits::chunk_ptr_ring();
        if(arena.begin(stream))
        {
            void const *ring_base = arena.upload_tables(tabs);
            if(ring_base != nullptr)
            {
                const T *const *src_tab = arena.table_view<const T *>(
                    ring_base, o_src_ptrs);
                T *const *dst_tab = arena.table_view<T *>(
                    ring_base, o_dst_ptrs);
                kernel::transpose::chunked<T>(
                    stream,
                    args->m,
                    args->n,
                    args->alpha,
                    src_tab,
                    Index(cv_src.nchunks),
                    dst_tab,
                    Index(cv_dst.nchunks),
                    Index(cv_src.chunk_elems),
                    Index(cv_dst.chunk_elems));
                arena.end(stream);
                return;
            }
        }
    }
        // (rule 11, user directive 2026-10-19) There is no per-piece
    // fallback: the 16 MiB pow2 chunk class and the pow2 2^20-slot ring
    // serve any launch up to a 16 TiB operand, so a failed upload is an
    // invariant violation, not a mode to stage around.
    std::fprintf(stderr,
        "nntile_transpose: chunk-ptr ring cannot serve this launch (invariant "
        "violation; the per-piece staging fallback was deleted by rule "
        "11)\n");
    std::abort();
}


// Specializations of CUDA wrapper for accelerated types
template<>
void Transpose<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Transpose<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Transpose<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for transpose tasks
template<typename T>
std::uint64_t Transpose<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    return hash;
}

template<typename T>
void Transpose<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Scalar alpha, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
//! Insert transpose task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->alpha = alpha;
    // Put amount of read-write bytes into flop count
    double nflops = sizeof(T) * 2 * m * n;
    // Degenerate-geometry guard (family-2 follow-up A): the panel driver
    // emits one sub-call per (src column window) x (dst row batch). In
    // the healthy chunk classes the windows batch many columns/rows and
    // the launch count is tiny, but a chunk at or below n dst-row bytes
    // (chunk_elems <= n) holds at most one dst row, so every window
    // degrades to one launch PER ELEMENT (3072x3072 fp32 at 12288-byte
    // chunks = 9.4M launches, whose enqueue alone stalls the runtime
    // watchdog; the policy ladder never picks such a class for a real
    // tile, but the driver must survive one). Estimate the launch count
    // with the driver's own worst-case arithmetic - W = ceil(n / P) src
    // windows of P = max(1, chunk_elems/m) columns, each refined into
    // B = ceil(m / rows_per_chunk) row batches with
    // rows_per_chunk = max(1, chunk_elems/n) - and route the submission
    // through the staging fallback when it exceeds 1 Mi launches.
    // Threshold rationale: the family-2 torture geometries (m=1000,
    // n=777 at 12288-byte chunks ~ 86K launches) complete in well under
    // a second and are pinned NATIVE by their tests, so the guard must
    // sit at least an order of magnitude above them; the degenerate
    // regime that motivated the guard measures 9.4M (~30 s of enqueue
    // before the watchdog kills it), so the guard must sit an order of
    // magnitude below that. Staging itself costs two buffer-sized
    // copies plus one classic kernel. Unchunked or healthy-chunked
    // submissions keep the native panel body, bitwise either way.
    (void)nflops;
    bool degenerate = false;
    std::size_t cb = 0;
    if(stage::handle_is_chunked(src) || stage::handle_is_chunked(dst))
    {
        // The REAL chunk class when the handle announced itself with its
        // size (tiles and the test factories), the policy replay below
        // otherwise - a wrong estimate here either needlesssly stages a
        // healthy geometry or, worse, lets the 9.4M-launch regime
        // through. Take the MIN over BOTH announced classes (0 = not
        // chunked = +inf): a mismatched-class pair must not bypass the
        // guard through the healthier operand's class.
        std::size_t const cb_dst = stage::handle_chunk_bytes(dst);
        cb = stage::handle_chunk_bytes(src);
        if(cb == 0 || (cb_dst != 0 && cb_dst < cb))
        {
            cb = cb_dst;
        }
        std::size_t const chunk_elems = cb >= sizeof(T) ? cb / sizeof(T) : 1;
        std::size_t const p_cols = chunk_elems / std::size_t(m > 0 ? m : 1);
        std::size_t const P = p_cols < std::size_t(n > 0 ? n : 1)
            ? (p_cols > 0 ? p_cols : 1)
            : std::size_t(n > 0 ? n : 1);
        std::size_t const rows_per_chunk =
            chunk_elems / std::size_t(n > 0 ? n : 1);
        std::size_t const B = (rows_per_chunk > 0
                ? (std::size_t(m > 0 ? m : 1) + rows_per_chunk - 1)
                    / rows_per_chunk
                : std::size_t(m > 0 ? m : 1));
        std::size_t const W =
            (std::size_t(n > 0 ? n : 1) + P - 1) / P;
        degenerate = W * B > (std::size_t{1} << 20);
    }
    if(degenerate)
    {
        // The law is pointer math everywhere outside GEMM's workspace;
        // this valve exists only so a pathological chunk class cannot
        // stall the watchdog. The env probe makes any actual use
        // visible in a run's log (a real geometry never trips it).
        if (std::getenv("NNHAUL_TRANSPOSE_STATS") != nullptr)
        {
            std::fprintf(stderr,
                "nntile_transpose: STAGED degenerate m=%lld n=%lld "
                "chunk_bytes=%zu (the pointer-math law's only valve)\n",
                static_cast<long long>(m), static_cast<long long>(n),
                cb);
        }
        ::nntile::haul::stage::staged_insert_task(codelet.raw,
            starpu_worker_hint,
            { { STARPU_R, &src }, { STARPU_W, &dst } }, args,
            sizeof(*args));
    }
    else
    {
        // Submit task: the BODY picks the unchunked fast path or the
        // native chunked panel driver.
        ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
            { { STARPU_R, &src }, { STARPU_W, &dst } }, args,
            sizeof(*args));
    }
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Transpose<std::tuple<nntile::fp64_t>>;
template class Transpose<std::tuple<nntile::fp32_t>>;
template class Transpose<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Transpose<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Transpose<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Transpose<std::tuple<nntile::bf16_t>>;
template class Transpose<std::tuple<nntile::fp16_t>>;

//! Pack of transpose operations for different types
transpose_pack_t transpose;

} // namespace nntile::haul
