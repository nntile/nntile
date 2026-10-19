/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/multiply_slice.cc
 * StarPU wrappers for per-element product of a tensor and a broadcasted slice
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/multiply_slice.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/multiply_slice.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked multiply_slice driver (chunk_units.hh scheme; see the
// index-map derivation in add_slice.cc): dst[i,l,j] = alpha * dst[i,l,j] *
// src[i,j] - ELEMENTWISE given the src slice value of the unit, so
// whole-row pieces reproduce the plain call's per-element operations
// bitwise. Unlike add_slice the tensor operand is a single RW buffer: an
// in-chunk row is transformed IN PLACE by the kernel (the plain call's own
// contract), and a row straddling a dst chunk boundary is gathered into a
// bounded m-element temp, transformed as (m, 1, 1) and scattered back; the
// src read of a piece is its panel's m-element segment, staged through a
// bounded temp when it straddles. Zero new kernels; all-unchunked buffers
// keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void multiply_slice_chunked(Index m, Index n, Index k, Scalar alpha,
    ::nnhaul::Buffer *buf_src, ::nnhaul::Buffer *buf_dst,
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
                // The kernel READS dst (in-place product): pull the
                // straddling row's current bytes into the temp first.
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
                sub(m, len / k, k, alpha, ps, pd);
            }
            else
            {
                sub(m, Index{1}, len, alpha, ps, pd);
            }
            if(dst_str && arena.ok_state())
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
MultiplySlice<std::tuple<T>>::MultiplySlice():
    codelet(
        "nntile_multiply_slice",
        &MultiplySlice<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &MultiplySlice<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &MultiplySlice<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::multiply_slice::cpu<T>
template<typename T>
void MultiplySlice<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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
        chunkunits::classic_dispatch_tripwire("nntile_multiply_slice", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::multiply_slice::cpu<T>(args->m, args->n, args->k,
            args->alpha, buf_src->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        multiply_slice_chunked<T>(
            args->m,
            args->n,
            args->k,
            args->alpha,
            buf_src,
            buf_dst,
            arena,
            [](Index mm, Index nn, Index kk, Scalar al, T const *ps, T *pd) {
                kernel::multiply_slice::cpu<T>(mm, nn, kk, al, ps, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_multiply_slice: cpu chunked driver aborted after an "
            "allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void MultiplySlice<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplySlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MultiplySlice<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplySlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MultiplySlice<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplySlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::multiply_slice::cuda<T>
template<typename T>
void MultiplySlice<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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
        chunkunits::classic_dispatch_tripwire("nntile_multiply_slice", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::multiply_slice::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, buf_src->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    // Chunked single-launch path: issue the WHOLE region as ONE chunk-aware
    // kernel instead of the per-piece driver's one launch per chunk-resident
    // panel run plus a gather/scatter staging pair for every row that
    // straddles a chunk boundary. The chunk base-pointer tables are built on
    // the host and parked in the shared device chunk-ptr ring (no chunk-count cap); the kernel stages them into
    // shared memory per block. Elementwise and bitwise the whole-buffer call.
    // The per-piece driver below stays as the fallback for a
    // test hook and a failed table allocation.
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
        kernel::multiply_slice::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, buf_src->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    if(args->m > 0 && args->n > 0 && args->k > 0)
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
                const T *const *ts = arena.table_view<const T *>(
                    ring_base, o_src_ptrs);
                T *const *td = arena.table_view<T *>(ring_base, o_dst_ptrs);
                kernel::multiply_slice::chunked<T>(
                    stream,
                    args->m,
                    args->n,
                    args->k,
                    args->alpha,
                    ts,
                    Index(cv_src.nchunks),
                    Index(cv_src.chunk_elems),
                    td,
                    Index(cv_dst.nchunks),
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
        "nntile_multiply_slice: chunk-ptr ring cannot serve this launch (invariant "
        "violation; the per-piece staging fallback was deleted by rule "
        "11)\n");
    std::abort();
}


// Specializations of CUDA wrapper for accelerated types
template<>
void MultiplySlice<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplySlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void MultiplySlice<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplySlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void MultiplySlice<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplySlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for multiply_slice tasks
template<typename T>
std::uint64_t MultiplySlice<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, n and k
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    return hash;
}

//! Submit multiply_slice task
template<typename T>
void MultiplySlice<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, Scalar alpha, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
//! Insert multiply_slice task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->alpha = alpha;
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * m * (2*k+1) * n;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked unit driver; multiply_slice never stages.
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_RW, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class MultiplySlice<std::tuple<nntile::fp64_t>>;
template class MultiplySlice<std::tuple<nntile::fp32_t>>;
template class MultiplySlice<std::tuple<nntile::fp32_fast_tf32_t>>;
template class MultiplySlice<std::tuple<nntile::fp32_fast_fp16_t>>;
template class MultiplySlice<std::tuple<nntile::fp32_fast_bf16_t>>;
template class MultiplySlice<std::tuple<nntile::bf16_t>>;
template class MultiplySlice<std::tuple<nntile::fp16_t>>;

//! Pack of multiply_slice operations for different types
multiply_slice_pack_t multiply_slice;

} // namespace nntile::haul
