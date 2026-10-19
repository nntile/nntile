/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/add_slice_inplace.cc
 * StarPU wrappers for addition of a tensor and a broadcasted slice
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/add_slice_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/add_slice_inplace.hh"
#include "nntile/nnhaul/chunk_units.hh"
#include "nntile/nnhaul/ops/add_inplace.hh"
#include "nntile/nnhaul/ops/scale_inplace.hh"
#include "nntile/nnhaul/ops/scale_slice.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked add_slice_inplace driver (chunk_units.hh scheme; see
// the index-map derivation in add_slice.cc - the op is ELEMENTWISE given
// the src slice value of the unit, so whole-row pieces reproduce the
// plain call's per-element operations bitwise). The dst tile is
// read-write: a row straddling a dst chunk boundary is gathered into a
// bounded m-element temp, transformed in place as (m, 1, 1) and
// scattered back; the src read of a piece is its panel's m-element
// segment, staged through a bounded temp when it straddles. Zero new
// kernels; all-unchunked buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void add_slice_inplace_chunked(Index m, Index n, Index k, Scalar alpha,
    ::nnhaul::Buffer *buf_src, Scalar beta, ::nnhaul::Buffer *buf_dst,
    Arena &arena, Sub &&sub)
{
    if(m <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_src{&cv_src, std::size_t(m), k};
    bool const chunked_dst = cv_dst.nchunks > 1;
    T *temp_dst = chunked_dst ? arena.alloc(std::size_t(m)) : nullptr;
    chunkunits::SliceStage<T, Arena> src_stage(
        cv_src, std::size_t(m), false, arena);
    chunkunits::walk_unit_runs(n, k,
        [&](Index i, Index len, bool /*first*/, bool dst_str,
            bool /*src_str*/)
        {
            if(len <= 0 || !arena.ok_state())
            {
                return;
            }
            T *pd = nullptr;
            if(dst_str)
            {
                chunkunits::gather_range(cv_dst, std::size_t(m) * i,
                    std::size_t(m), temp_dst, arena);
                pd = temp_dst;
            }
            else
            {
                pd = span_dst.ptr(i);
            }
            T const *ps = src_stage.advance(i / k, false);
            if(pd == nullptr || ps == nullptr || !arena.ok_state())
            {
                return;
            }
            if(len >= k)
            {
                sub(m, len / k, k, alpha, ps, beta, pd);
            }
            else
            {
                sub(m, Index{1}, len, alpha, ps, beta, pd);
            }
            if(dst_str)
            {
                chunkunits::scatter_range(cv_dst, std::size_t(m) * i,
                    std::size_t(m), temp_dst, arena);
            }
        },
        span_dst, span_src);
}

} // namespace

//! Constructor
template<typename T>
AddSliceInplace<std::tuple<T>>::AddSliceInplace():
    codelet(
        "nntile_add_slice_inplace",
        &AddSliceInplace<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &AddSliceInplace<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &AddSliceInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::add_slice_inplace::cpu<T>
template<typename T>
void AddSliceInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
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
        chunkunits::classic_dispatch_tripwire("nntile_add_slice_inplace", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::add_slice_inplace::cpu<T>(args->m, args->n, args->k,
            args->alpha, buf_src->get_ptr<T>(), args->beta,
            buf_dst->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        add_slice_inplace_chunked<T>(
            args->m,
            args->n,
            args->k,
            args->alpha,
            buf_src,
            args->beta,
            buf_dst,
            arena,
            [](Index mm, Index nn, Index kk, Scalar al, T const *ps,
                Scalar be, T *pd) {
                kernel::add_slice_inplace::cpu<T>(mm, nn, kk, al, ps, be, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_add_slice_inplace: cpu chunked driver aborted after an "
            "allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void AddSliceInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSliceInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddSliceInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSliceInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddSliceInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSliceInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! NNHaul wrapper for kernel::add_slice_inplace::cuda<T>
template<typename T>
void AddSliceInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
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
        chunkunits::classic_dispatch_tripwire("nntile_add_slice_inplace", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::add_slice_inplace::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, buf_src->get_ptr<T>(), args->beta,
            buf_dst->get_ptr<T>());
        return;
    }
    // Chunked single-launch path: ONE chunk-aware kernel over the whole
    // (m, n, k) range instead of the per-piece driver's launches. The chunk
    // base-pointer tables are built on the host and parked in the shared
    // device chunk-ptr ring (no chunk-count cap); the
    // kernel reads the table per-thread by shift/mask (rule 10: no smem staging). Elements are
    // independent and the expression (including the m == 1 case) is
    // identical, so the result is cuda()'s bitwise. The piece driver below
    // stays as the fallback for a failed table
    // allocation.
    // Piece C (2026-10-18c): the all-1-chunk degenerate launch is
    // the FIRST thing the chunked block does -- the chunk counts are
    // read straight off the buffers, before any
    // ChunkView/UnitSpan/RingTables work exists, so the degenerate
    // host path is as lean as the classic dispatch above (gemm
    // path-F precedent; launch pointer-identical, bitwise by
    // construction, ring untouched).
    if(args->m > 0 && args->n > 0 && args->k > 0
        && buf_src->chunk_count() == 1
        && buf_dst->chunk_count() == 1)
    {
        kernel::add_slice_inplace::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, buf_src->get_ptr<T>(), args->beta,
            buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    if(args->m > 0 && args->n > 0 && args->k > 0)
    {
        chunkunits::RingTables &tabs = chunkunits::ring_tables();
        tabs.clear();
        std::size_t const o_ps = tabs.add(cv_src);
        std::size_t const o_pd = tabs.add(cv_dst);
        chunkunits::ChunkPtrRing &arena = chunkunits::chunk_ptr_ring();
        if(arena.begin(stream))
        {
            void const *ring_base = arena.upload_tables(tabs);
            if(ring_base != nullptr)
            {
                const T *const *ts = arena.table_view<const T *>(
                    ring_base, o_ps);
                T *const *td = arena.table_view<T *>(ring_base, o_pd);
                kernel::add_slice_inplace::chunked<T>(
                    stream,
                    args->m,
                    args->n,
                    args->k,
                    args->alpha,
                    ts,
                    Index(cv_src.nchunks),
                    args->beta,
                    td,
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
        "nntile_add_slice_inplace: chunk-ptr ring cannot serve this launch (invariant "
        "violation; the per-piece staging fallback was deleted by rule "
        "11)\n");
    std::abort();
}


// Specializations of CUDA wrapper for accelerated types
template<>
void AddSliceInplace<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSliceInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AddSliceInplace<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSliceInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AddSliceInplace<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSliceInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add_slice_inplace tasks
template<typename T>
std::uint64_t AddSliceInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void AddSliceInplace<std::tuple<T>>::submit(int starpu_worker_hint,
    Index m,
    Index n,
    Index k,
    Scalar alpha,
    ::nnhaul::Handle & src,
    Scalar beta,
    ::nnhaul::Handle & dst
)
//! Insert add_slice_inplace task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // If k is 1, then this operation reduces to add_inplace
    if(k == 1)
    {
        add_inplace.submit<std::tuple<T>>(starpu_worker_hint, m*n, alpha, src, beta, dst);
        return;
    }
    // If alpha is zero then reduce to scale_inplace
    if(alpha == 0.0)
    {
        scale_inplace.submit<std::tuple<T>>(starpu_worker_hint, m*n*k, beta, dst);
        return;
    }
    // If beta is zero then reduce to scale_slice
    if(beta == 0.0)
    {
        scale_slice.submit<std::tuple<T>>(starpu_worker_hint, m, n, k, alpha, src, dst);
        return;
    }
    // Access mode for the dst handle
    enum starpu_data_access_mode dst_mode;
    if(beta == 1.0)
    {
        dst_mode = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);
    }
    else
    {
        dst_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t*)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->alpha = alpha;
    args->beta = beta;
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * m * (2*k+1) * n;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked unit driver; add_slice_inplace never stages.
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AddSliceInplace<std::tuple<nntile::fp64_t>>;
template class AddSliceInplace<std::tuple<nntile::fp32_t>>;
template class AddSliceInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AddSliceInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AddSliceInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AddSliceInplace<std::tuple<nntile::bf16_t>>;
template class AddSliceInplace<std::tuple<nntile::fp16_t>>;

//! Pack of add_slice_inplace operations for different types
add_slice_inplace_pack_t add_slice_inplace;

} // namespace nntile::haul
