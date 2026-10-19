/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/norm_slice_inplace.cc
 * Euclidean norms of fibers into a slice of a StarPU buffer
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/norm_slice_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/norm_slice_inplace.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked norm_slice_inplace driver (chunk_units.hh scheme; the
// layout derivation and the piecewise-hypot combine are documented in
// norm_slice.cc - first piece (alpha, caller beta) against the real dst
// slice, continuation pieces (alpha, 1) accumulating in place, exact up
// to the piecewise rounding). All-zero pieces anchor the chain because
// the CUDA m>1 kernel writes the beta-combined value for all-zero
// fibers (0 for beta==0, fabs(beta*dst_old) otherwise), matching the
// CPU kernel and the CUDA m==1 kernel. The dst slice segment of a panel
// is used in place or staged through a bounded m temp for the panel's
// chain (gathered when the caller's beta != 0, zero-initialized
// otherwise, scattered back at the panel switch); rows straddling src
// chunk boundaries are gathered into a bounded m temp and reduced as
// (m, 1, 1) pieces. Panel-aligned pieces may span whole panels. Zero
// new kernels; all-unchunked buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void norm_slice_inplace_chunked(Index m, Index n, Index k, Scalar alpha,
    ::nnhaul::Buffer *buf_src, Scalar beta, ::nnhaul::Buffer *buf_dst,
    Arena &arena, Sub &&sub)
{
    if(m <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    chunkunits::UnitSpan<T> const span_src{&cv_src, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t(m), k};
    bool const chunked_src = cv_src.nchunks > 1;
    T *temp_src = chunked_src ? arena.alloc(std::size_t(m)) : nullptr;
    chunkunits::SliceStage<T, Arena> dst_stage(
        cv_dst, std::size_t(m), true, arena);
    chunkunits::walk_unit_runs(n, k,
        [&](Index i, Index len, bool first, bool src_str, bool /*dst_str*/)
        {
            if(len <= 0 || !arena.ok_state())
            {
                return;
            }
            Index const panel = i / k;
            T *pd = dst_stage.advance(panel, first && beta != Scalar{0});
            T const *ps = nullptr;
            if(src_str)
            {
                chunkunits::gather_range(cv_src, std::size_t(m) * i,
                    std::size_t(m), temp_src, arena);
                ps = temp_src;
            }
            else
            {
                ps = span_src.ptr(i);
            }
            if(ps == nullptr || pd == nullptr || !arena.ok_state())
            {
                return;
            }
            Scalar const beta_eff = first ? beta : Scalar{1};
            if(len >= k)
            {
                sub(m, len / k, k, ps, beta_eff, pd);
            }
            else
            {
                sub(m, Index{1}, len, ps, beta_eff, pd);
            }
        },
        span_src, span_dst);
    dst_stage.retire();
}

} // namespace

//! Constructor
template<typename T>
NormSliceInplace<std::tuple<T>>::NormSliceInplace():
    codelet(
        "nntile_norm_slice_inplace",
        &NormSliceInplace<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &NormSliceInplace<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &NormSliceInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::norm_slice_inplace::norm_slice_inplace<T>
template<typename T>
void NormSliceInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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
        chunkunits::classic_dispatch_tripwire("nntile_norm_slice_inplace", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::norm_slice_inplace::cpu<T>(args->m, args->n, args->k,
            args->alpha, buf_src->get_ptr<T>(), args->beta,
            buf_dst->get_ptr<T>());
        return;
    }
    Scalar const alpha = args->alpha;
    try
    {
        chunkunits::ArenaCpu<T> arena;
        norm_slice_inplace_chunked<T>(
            args->m,
            args->n,
            args->k,
            alpha,
            buf_src,
            args->beta,
            buf_dst,
            arena,
            [alpha](Index mm, Index nn, Index kk, T const *ps, Scalar be,
                T *pd) {
                kernel::norm_slice_inplace::cpu<T>(mm, nn, kk, alpha, ps, be,
                    pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_norm_slice_inplace: cpu chunked driver aborted after "
            "an allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void NormSliceInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSliceInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void NormSliceInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSliceInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void NormSliceInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSliceInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! NNHaul wrapper for kernel::norm_slice_inplace::norm_slice_inplace<T>
template<typename T>
void NormSliceInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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
        chunkunits::classic_dispatch_tripwire("nntile_norm_slice_inplace", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::norm_slice_inplace::cuda<T>(stream, args->m, args->n,
            args->k, args->alpha, buf_src->get_ptr<T>(), args->beta,
            buf_dst->get_ptr<T>());
        return;
    }
    // Chunked single-launch path (see sum_slice.cc for the rationale): one
    // chunk-aware kernel over the whole logical range; the chunk base-pointer
    // tables are built on the host and parked in the shared device chunk-ptr ring's
    // device-resident ring (no chunk-count cap), and the kernel stages
    // them into shared memory per block. The piece driver stays as the
    // fallback for a failed table allocation.
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
        kernel::norm_slice_inplace::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, buf_src->get_ptr<T>(), args->beta, buf_dst->get_ptr<T>());
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
                kernel::norm_slice_inplace::chunked<T>(
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
        "nntile_norm_slice_inplace: chunk-ptr ring cannot serve this launch (invariant "
        "violation; the per-piece staging fallback was deleted by rule "
        "11)\n");
    std::abort();
}


// Specializations of CUDA wrapper for accelerated types
template<>
void NormSliceInplace<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSliceInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void NormSliceInplace<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSliceInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void NormSliceInplace<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSliceInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for norm_slice_inplace tasks
template<typename T>
std::uint64_t NormSliceInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
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

template<typename T>
void NormSliceInplace<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, Scalar alpha, ::nnhaul::Handle & src, Scalar beta,
        ::nnhaul::Handle & dst, int redux)
//! Insert norm_slice_inplace task into StarPU pool of tasks
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
    args->alpha = alpha;
    args->beta = beta;
    // Put amount of bytes read and write inplace of gflops
    size_t src_nbytes = sizeof(T) * m * k * n;
    size_t dst_nbytes = sizeof(T) * m * n;
    double nflops = beta == 0.0 ? src_nbytes + dst_nbytes :
        src_nbytes + 2*dst_nbytes;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked unit driver; norm_slice_inplace never stages.
    (void)nflops;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class NormSliceInplace<std::tuple<nntile::fp64_t>>;
template class NormSliceInplace<std::tuple<nntile::fp32_t>>;
template class NormSliceInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class NormSliceInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class NormSliceInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class NormSliceInplace<std::tuple<nntile::bf16_t>>;
template class NormSliceInplace<std::tuple<nntile::fp16_t>>;

//! Pack of norm_slice_inplace operations for different types
norm_slice_inplace_pack_t norm_slice_inplace;

} // namespace nntile::haul
