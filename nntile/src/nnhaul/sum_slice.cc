/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/sum_slice.cc
 * Sum over fibers into a slice of a NNHaul buffer
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/sum_slice.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/sum_slice.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::nnhaul
{

// ---------------------------------------------------------------------------
// Native chunked sum_slice driver (chunk_units.hh scheme; the layout
// derivation is the maxsumexp one - src [m, k, n] with m contiguous, the
// fiber of unit (i0, i2) taking one element per row of the panel). The
// pieces are runs of whole rows inside single src chunks; a piece
// computes the PARTIAL Kahan sum of every unit of its panel and the
// pieces of a panel chain through the kernel's OWN beta!=0 accumulate
// contract: first piece (alpha, caller beta), continuations (alpha, 1) -
// algebraically exact, applied in fiber-position order on the single
// in-order stream, equal to the plain call up to the rounding of the
// piecewise merge (the CUDA kernel adds its shared-memory atomicAdd
// nondeterminism, which the plain call already has). The dst slice
// segment of a panel (m elements) is used in place or staged through a
// bounded m temp for the panel's chain (gathered when the caller's beta
// != 0, zero-initialized otherwise, scattered back at the panel switch);
// rows straddling src chunk boundaries are gathered into a bounded m
// temp and reduced as (m, 1, 1) pieces. Panel-aligned pieces may span
// whole panels. Zero new kernels; all-unchunked buffers keep the single
// classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void sum_slice_chunked(Index m, Index n, Index k, Scalar alpha,
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
            if(len >= k)
            {
                sub(m, len / k, k, ps, alpha, first ? beta : Scalar{1}, pd);
            }
            else
            {
                sub(m, Index{1}, len, ps, alpha, first ? beta : Scalar{1},
                    pd);
            }
        },
        span_src, span_dst);
    dst_stage.retire();
}

} // namespace

//! Constructor
template<typename T>
SumSlice<std::tuple<T>>::SumSlice():
    codelet(
        "nntile_sum_slice",
        &SumSlice<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &SumSlice<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &SumSlice<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::sum_slice::cpu<T>
template<typename T>
void SumSlice<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    if((buf_src->chunk_count() == 1) && (buf_dst->chunk_count() == 1)
       )
    {
        kernel::sum_slice::cpu<T>(args->m, args->n, args->k, args->alpha,
            buf_src->get_ptr<T>(), args->beta, buf_dst->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        sum_slice_chunked<T>(
            args->m,
            args->n,
            args->k,
            args->alpha,
            buf_src,
            args->beta,
            buf_dst,
            arena,
            [](Index mm, Index nn, Index kk, T const *ps, Scalar al,
                Scalar be, T *pd) {
                kernel::sum_slice::cpu<T>(mm, nn, kk, al, ps, be, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_sum_slice: cpu chunked driver aborted after an "
            "allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void SumSlice<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SumSlice<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SumSlice<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! NNHaul wrapper for kernel::sum_slice::cuda<T>
template<typename T>
void SumSlice<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if((buf_src->chunk_count() == 1) && (buf_dst->chunk_count() == 1)
       )
    {
        kernel::sum_slice::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, buf_src->get_ptr<T>(), args->beta,
            buf_dst->get_ptr<T>());
        return;
    }
        // Chunked single-launch path: the whole multi-chunk range is
        // issued as ONE chunk-aware kernel reading the shared device
    // chunk-pointer table; the per-element math matches cuda().
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    if(args->m > 0 && args->n > 0 && args->k > 0)
    {
            chunkunits::LaunchTables &tabs = chunkunits::launch_tables();
        tabs.clear();
        std::size_t const o_src_ptrs = tabs.add(cv_src);
        std::size_t const o_dst_ptrs = tabs.add(cv_dst);
        chunkunits::ChunkPtrTable &arena = chunkunits::chunk_ptr_table();
        if(arena.begin(stream))
        {
            void const *table_base = arena.upload_tables(tabs);
            if(table_base != nullptr)
            {
                const T *const *ts = arena.table_view<const T *>(
                    table_base, o_src_ptrs);
                T *const *td = arena.table_view<T *>(table_base, o_dst_ptrs);
                kernel::sum_slice::chunked<T>(
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
        // There is no per-piece
    // fallback: the 16 MiB pow2 chunk class and the pow2 2^20-slot table
    // serve any launch up to a 16 TiB operand, so a failed upload is an
    // invariant violation, not a mode to stage around.
    std::fprintf(stderr,
        "nntile_sum_slice: chunk-ptr table cannot serve this launch (invariant "
        "violation; no per-piece staging fallback exists)\n");
    std::abort();
}

// Specializations of CUDA wrapper for accelerated types
template<>
void SumSlice<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SumSlice<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SumSlice<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for sum_slice tasks
template<typename T>
std::uint64_t SumSlice<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, n and k
    uint32_t hash = 0;
    hash = ::nntile::nnhaul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::nnhaul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::nnhaul::fnv1a(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void SumSlice<std::tuple<T>>::submit(int worker_hint, Index m, Index n, Index k, Scalar alpha, ::nnhaul::Handle & src, Scalar beta,
        ::nnhaul::Handle & dst, int redux)
//! Insert sum_slice task into NNHaul task pool
/*! No argument checking is performed. All the inputs are packed and passed to
 * submit() call. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Access mode for the dst handle
    constexpr Scalar zero = 0, one = 1;
    enum data_access_mode dst_mode;
    if(beta == zero)
    {
        dst_mode = ACCESS_W;
    }
    else if(beta == one)
    {
        (void)redux;
        dst_mode = static_cast<data_access_mode>(ACCESS_RW | ACCESS_COMMUTE);
    }
    else
    {
        dst_mode = ACCESS_RW;
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
    // chunked unit driver; sum_slice never stages.
    (void)nflops;
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_R, &src }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class SumSlice<std::tuple<nntile::fp64_t>>;
template class SumSlice<std::tuple<nntile::fp32_t>>;
template class SumSlice<std::tuple<nntile::fp32_fast_tf32_t>>;
template class SumSlice<std::tuple<nntile::fp32_fast_fp16_t>>;
template class SumSlice<std::tuple<nntile::fp32_fast_bf16_t>>;
template class SumSlice<std::tuple<nntile::bf16_t>>;
template class SumSlice<std::tuple<nntile::fp16_t>>;

//! Pack of sum_slice operations for different types
sum_slice_pack_t sum_slice;

} // namespace nntile::nnhaul
