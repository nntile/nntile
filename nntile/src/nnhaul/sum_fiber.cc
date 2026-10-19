/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/sum_fiber.cc
 * Sums over slices into a fiber of a StarPU buffer
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/sum_fiber.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/sum_fiber.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked sum_fiber driver (chunk_units.hh fiber scheme, cf. the
// sum_slice reduction driver).
//
// Kernel index map (derived from kernel::sum_fiber::cpu/cuda, NOT
// assumed): src is [m, k, n, batch] with element (i0, i2, i1, b) at
// flat i0 + m*(i2 + k*(i1 + n*b)) - the contiguous unit is the
// m-element fiber, fibers group into k-fiber planes per (i1, b), and
// the output dst is the k-by-batch fiber dst[i2 + k*b]: one
// accumulator per (i2, b), reduced over the n planes' fibers
// (i2, i1, b), i1 = 0..n-1. Pieces are runs of whole fibers inside
// single src chunks (walk_unit_runs, panels = batches);
// split_fiber_piece re-cuts them into plane/batch sub-pieces that
// compute the PARTIAL Kahan sum of every (i2, b) they cover and chain
// through the kernel's OWN beta!=0 accumulate contract: a sub-piece
// starting in plane 0 of its batch (every accumulator's first
// contribution comes from its plane-0 fiber) takes the caller's
// (alpha, beta), every continuation takes (alpha, 1) - algebraically
// exact, applied in fiber-position order on the single in-order
// stream, equal to the plain call up to the rounding of the piecewise
// merge (the CUDA kernel adds its shared-memory atomicAdd
// nondeterminism, which the plain call already has). The dst batch
// segment ([k*b, k*b+k)) is used in place or staged through a bounded
// k temp for the batch's whole chain (gathered when the caller's beta
// != 0, zero-initialized otherwise - the kernel's beta==0 branch
// never reads dst); sub-pieces address their sub-range by adding the
// in-plane fiber offset. Straddling src fibers are gathered into a
// bounded m temp and reduced as (m, 1, 1, 1) pieces. Zero new kernels;
// all-unchunked buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void sum_fiber_chunked(Index m, Index n, Index k, Index batch, Scalar alpha,
    ::nnhaul::Buffer *buf_src, Scalar beta, ::nnhaul::Buffer *buf_dst,
    Arena &arena, Sub &&sub)
{
    if(m <= 0 || k <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    chunkunits::UnitSpan<T> const span_src{&cv_src, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t(k), k * n};
    bool const chunked_src = cv_src.nchunks > 1;
    T *temp_src = chunked_src ? arena.alloc(std::size_t(m)) : nullptr;
    chunkunits::SliceStage<T, Arena> dst_stage(
        cv_dst, std::size_t(k), true, arena);
    chunkunits::walk_unit_runs(batch, k * n,
        [&](Index i, Index len, bool /*first*/, bool src_str,
            bool /*dst_str*/)
        {
            if(len <= 0 || !arena.ok_state())
            {
                return;
            }
            chunkunits::split_fiber_piece(i, len, n, k,
                [&](Index j, Index L, bool plane_span, bool first_b)
                {
                    if(L <= 0 || !arena.ok_state())
                    {
                        return;
                    }
                    Index const b = j / (n * k);
                    T *pd = dst_stage.advance(b, beta != Scalar{0})
                        + (plane_span ? 0 : j % k);
                    T const *ps = nullptr;
                    if(src_str)
                    {
                        chunkunits::gather_range(cv_src, std::size_t(m) * j,
                            std::size_t(m), temp_src, arena);
                        ps = temp_src;
                    }
                    else
                    {
                        ps = span_src.ptr(j);
                    }
                    if(ps == nullptr || pd == nullptr || !arena.ok_state())
                    {
                        return;
                    }
                    Scalar const beta_eff = first_b ? beta : Scalar{1};
                    if(plane_span)
                    {
                        sub(m, L / k, k, ps, alpha, beta_eff, pd);
                    }
                    else
                    {
                        sub(m, Index{1}, L, ps, alpha, beta_eff, pd);
                    }
                });
        },
        span_src, span_dst);
    dst_stage.retire();
}

} // namespace

//! Constructor
template<typename T>
SumFiber<std::tuple<T>>::SumFiber():
    codelet(
        "nntile_sum_fiber",
        &SumFiber<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &SumFiber<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &SumFiber<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::sum_fiber::cpu<T>
template<typename T>
void SumFiber<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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
        chunkunits::classic_dispatch_tripwire("nntile_sum_fiber", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::sum_fiber::cpu<T>(args->m, args->n, args->k, args->batch,
            args->alpha, buf_src->get_ptr<T>(), args->beta,
            buf_dst->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        sum_fiber_chunked<T>(
            args->m,
            args->n,
            args->k,
            args->batch,
            args->alpha,
            buf_src,
            args->beta,
            buf_dst,
            arena,
            [](Index mm, Index nn, Index kk, T const *ps, Scalar al,
                Scalar be, T *pd) {
                kernel::sum_fiber::cpu<T>(mm, nn, kk, Index{1}, al, ps, be,
                    pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_sum_fiber: cpu chunked driver aborted after an "
            "allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void SumFiber<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SumFiber<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SumFiber<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::sum_fiber::cuda<T>
template<typename T>
void SumFiber<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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
        chunkunits::classic_dispatch_tripwire("nntile_sum_fiber", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::sum_fiber::cuda<T>(stream, args->m, args->n, args->k,
            args->batch, args->alpha, buf_src->get_ptr<T>(), args->beta,
            buf_dst->get_ptr<T>());
        return;
    }
    // Chunked single-launch path: the WHOLE reduction is issued as ONE
    // chunk-aware kernel with the two chunk base-pointer tables read from
    // the shared device chunk-ptr ring (no chunk-count cap), instead of the
    // per-piece driver's O(chunk-resident planes x straddles) launches. The
    // kernel keeps cuda()'s thread geometry and per-thread accumulation
    // order, so the result is cuda()'s up to the same atomic ordering
    // nondeterminism. A launch whose pointer set exceeds the ring, and the
    // arbitrary factorizations, keep the piece
    // driver below as the fallback.
    // Piece C (2026-10-18c): the all-1-chunk degenerate launch is
    // the FIRST thing the chunked block does -- the chunk counts are
    // read straight off the buffers, before any
    // ChunkView/UnitSpan/RingTables work exists, so the degenerate
    // host path is as lean as the classic dispatch above (gemm
    // path-F precedent; launch pointer-identical, bitwise by
    // construction, ring untouched).
    if(args->m > 0 && args->n > 0 && args->k > 0 && args->batch > 0
        && buf_src->chunk_count() == 1
        && buf_dst->chunk_count() == 1)
    {
        kernel::sum_fiber::cuda<T>(stream, args->m, args->n, args->k,
            args->batch, args->alpha, buf_src->get_ptr<T>(),
            args->beta, buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    if(args->m > 0 && args->n > 0 && args->k > 0 && args->batch > 0)
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
                kernel::sum_fiber::chunked<T>(
                    stream,
                    args->m,
                    args->n,
                    args->k,
                    args->batch,
                    args->alpha,
                    src_tab,
                    Index(cv_src.nchunks),
                    args->beta,
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
        "nntile_sum_fiber: chunk-ptr ring cannot serve this launch (invariant "
        "violation; the per-piece staging fallback was deleted by rule "
        "11)\n");
    std::abort();
}


// Specializations of CUDA wrapper for accelerated types
template<>
void SumFiber<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumFiber<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SumFiber<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumFiber<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SumFiber<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumFiber<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for sum_fiber tasks
template<typename T>
std::uint64_t SumFiber<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, n and k
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    hash = ::nntile::haul::fnv1a(&args->batch, sizeof(args->batch), hash);
    return hash;
}

template<typename T>
void SumFiber<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, Index batch,
        Scalar alpha, ::nnhaul::Handle & src,
        Scalar beta, ::nnhaul::Handle & dst, int redux)
//! Insert sum_fiber task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Access mode for the dst handle
    constexpr Scalar zero = 0, one = 1;
    enum starpu_data_access_mode dst_mode;
    if(beta == zero)
    {
        dst_mode = STARPU_W;
    }
    else if(beta == one)
    {
        if(redux != 0)
        {
            dst_mode = STARPU_REDUX;
        }
        else
        {
            dst_mode = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);
        }
    }
    else
    {
        dst_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->batch = batch;
    args->alpha = alpha;
    args->beta = beta;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked fiber driver; sum_fiber never stages.
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class SumFiber<std::tuple<nntile::fp64_t>>;
template class SumFiber<std::tuple<nntile::fp32_t>>;
template class SumFiber<std::tuple<nntile::fp32_fast_tf32_t>>;
template class SumFiber<std::tuple<nntile::fp32_fast_fp16_t>>;
template class SumFiber<std::tuple<nntile::fp32_fast_bf16_t>>;
template class SumFiber<std::tuple<nntile::bf16_t>>;
template class SumFiber<std::tuple<nntile::fp16_t>>;

//! Pack of sum_fiber operations for different types
sum_fiber_pack_t sum_fiber;

} // namespace nntile::haul
