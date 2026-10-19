/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/maxsumexp.cc
 * Max and sum of exponents for StarPU buffer
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/maxsumexp.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

// Other NNTile headers
#include "nntile/kernel/maxsumexp.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked maxsumexp driver (chunk_units.hh scheme, cf. gemm.cc)
//
// Kernel index map (derived from kernel::maxsumexp::cpu/cuda, NOT
// assumed): src is [m, k, n] with m CONTIGUOUS (element (i0, i1, i2) at
// flat i0 + m*i1 + m*k*i2) and the reduction fiber of the unit (i0, i2)
// is the k elements at flat i0 + m*i1 + m*k*i2 for i1 in [0, k) - i.e.
// ONE ELEMENT IN EACH ROW of the unit's panel of k consecutive rows; the
// (max, sumexp) pair lands at dst[2*(i0 + m*i2)].
//
// Decomposition: pieces are runs of whole rows (m contiguous elements)
// inside single src chunks. A piece computes PARTIAL (max, sumexp) of
// every unit of its panel over the fiber positions it covers, and the
// pieces of a panel are combined by the kernel's OWN accumulation
// contract (the beta!=0 merge): the first piece of each panel runs with
// the caller's beta, every continuation piece with beta=1, which merges
// max_first = max(max_a, max_b) and sum = sum_a*exp(max_a-max_first) +
// sum_b*exp(max_b-max_first) - the algebraically exact combine (the same
// algebra kernel::accumulate_maxsumexp implements for tensor-level
// partial merges), applied in fiber-position order on the single
// in-order stream. NO fiber is ever split mid-piece, so no extra combine
// kernel is needed; the result equals the plain call up to the rounding
// of the piecewise merge (CUDA adds its shared-memory atomicAdd
// nondeterminism, which the plain call already has).
//
// The dst slice segment of a panel (2*m elements) is used in place when
// it lies in one dst chunk, else staged through a bounded 2*m temp for
// the panel's whole piece chain (gathered when the caller's beta != 0,
// zero-initialized otherwise, scattered back at the panel switch). Rows
// straddling src chunk boundaries are gathered into a bounded m-element
// temp and reduced as (m, 1, 1) pieces. Panel-aligned pieces may span
// whole panels ((m, n', k) sub-calls, first-piece beta). Zero new
// kernels; all-unchunked buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void maxsumexp_chunked(Index m, Index n, Index k, ::nnhaul::Buffer *buf_src,
    ::nnhaul::Buffer *buf_dst, Scalar beta,
    Arena &arena, Sub &&sub)
{
    if(m <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    chunkunits::UnitSpan<T> const span_src{&cv_src, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t{2} * m, k};
    bool const chunked_src = cv_src.nchunks > 1;
    T *temp_src = chunked_src ? arena.alloc(std::size_t(m)) : nullptr;
    chunkunits::SliceStage<T, Arena> dst_stage(
        cv_dst, std::size_t{2} * m, true, arena);
    chunkunits::walk_unit_runs(n, k,
        [&](Index i, Index len, bool first, bool src_str,
            bool /*dst_str*/)
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
MaxSumExp<std::tuple<T>>::MaxSumExp():
    codelet(
        "nntile_maxsumexp",
        &MaxSumExp<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &MaxSumExp<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &MaxSumExp<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Max and sum of exponents along middle axis of NNHaul buffer on CPU
template<typename T>
void MaxSumExp<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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
        chunkunits::classic_dispatch_tripwire("nntile_maxsumexp", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::maxsumexp::cpu<T>(args->m, args->n, args->k,
            buf_src->get_ptr<T>(), args->beta, buf_dst->get_ptr<T>());
        return;
    }
    Scalar const beta = args->beta;
    try
    {
        chunkunits::ArenaCpu<T> arena;
        maxsumexp_chunked<T>(
            args->m,
            args->n,
            args->k,
            buf_src,
            buf_dst,
            beta,
            arena,
            [](Index mm, Index nn, Index kk, T const *ps, Scalar be, T *pd) {
                kernel::maxsumexp::cpu<T>(mm, nn, kk, ps, be, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_maxsumexp: cpu chunked driver aborted after an "
            "allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void MaxSumExp<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaxSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MaxSumExp<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaxSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MaxSumExp<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaxSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Max and sum of exponents along middle axis of NNHaul buffer on CUDA
template<typename T>
void MaxSumExp<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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
        chunkunits::classic_dispatch_tripwire("nntile_maxsumexp", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::maxsumexp::cuda<T>(stream, args->m, args->n, args->k,
            buf_src->get_ptr<T>(), args->beta, buf_dst->get_ptr<T>());
        return;
    }
    // Chunked single-launch path: ONE chunk-aware kernel over the whole
    // (m, n, k) range.  The piece driver below launches one kernel per
        // chunk-resident run and thereby pays the full m*n grid per piece; on
    // the GPT-2 attention shape (m=1, n=12288, k=1024) that dominated the
    // p90 tail.  The arithmetic order is unchanged, so this reproduces the
    // contiguous kernel up to its shared-atomic floor.  The piece driver
    // stays as the fallback for a failed
    // table upload.
    if(args->m > 0 && args->n > 0 && args->k > 0)
    {
        chunkunits::ChunkView<T> const cv_src(buf_src);
        chunkunits::ChunkView<T> const cv_dst(buf_dst);
        // Piece 7: the m==1 straddling panels are resolved INSIDE the one
        // launch now, so there is no host-side panel list and no third
        // table -- the merged upload is just (src | dst).
        // ONE merged table (src | dst) in the reusable thread-local
        // staging, ONE ring upload per launch.
        chunkunits::RingTables &tabs = chunkunits::ring_tables();
        tabs.clear();
        std::size_t const o_src = tabs.add(cv_src);
        std::size_t const o_dst = tabs.add(cv_dst);
        chunkunits::ChunkPtrRing &ring = chunkunits::chunk_ptr_ring();
        if(ring.begin(stream))
        {
            void const *ring_base = ring.upload_tables(tabs);
            if(ring_base != nullptr)
            {
                const T *const *tps = ring.table_view<const T *>(ring_base,
                    o_src);
                T *const *tpd = ring.table_view<T *>(ring_base, o_dst);
                kernel::maxsumexp::chunked<T>(
                    stream,
                    args->m,
                    args->n,
                    args->k,
                    tps,
                    Index(cv_src.nchunks),
                    args->beta,
                    tpd,
                    Index(cv_dst.nchunks),
                    Index(cv_src.chunk_elems),
                    Index(cv_dst.chunk_elems),
                    long(cv_src.total_elems),
                    long(cv_dst.total_elems));
                ring.end(stream);
                return;
            }
        }
    }
        // (rule 11, user directive 2026-10-19) There is no per-piece
    // fallback: the 16 MiB pow2 chunk class and the pow2 2^20-slot ring
    // serve any launch up to a 16 TiB operand, so a failed upload is an
    // invariant violation, not a mode to stage around.
    std::fprintf(stderr,
        "nntile_maxsumexp: chunk-ptr ring cannot serve this launch (invariant "
        "violation; the per-piece staging fallback was deleted by rule "
        "11)\n");
    std::abort();
}


// Specializations of CUDA wrapper for accelerated types
template<>
void MaxSumExp<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaxSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void MaxSumExp<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaxSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void MaxSumExp<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaxSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for maxsumexp tasks that depends only on m, n and k
template<typename T>
std::uint64_t MaxSumExp<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    hash = ::nntile::haul::fnv1a(&args->beta, sizeof(args->beta), hash);
    return hash;
}

template<typename T>
void MaxSumExp<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n,
        Index k, ::nnhaul::Handle & src, ::nnhaul::Handle & dst, Scalar beta, int redux)
//! Insert maxsumexp task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    constexpr Scalar zero = 0, one = 1;
    // Access mode for the dst handle
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
        throw std::runtime_error("maxsumexp: beta must be 0.0 or 1.0");
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->beta = beta;
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * m * (k+2) * n;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked unit driver; maxsumexp never goes through the staging
    // fallback.
    (void)nflops;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class MaxSumExp<std::tuple<nntile::fp64_t>>;
template class MaxSumExp<std::tuple<nntile::fp32_t>>;
template class MaxSumExp<std::tuple<nntile::fp32_fast_tf32_t>>;
template class MaxSumExp<std::tuple<nntile::fp32_fast_fp16_t>>;
template class MaxSumExp<std::tuple<nntile::fp32_fast_bf16_t>>;
template class MaxSumExp<std::tuple<nntile::bf16_t>>;
template class MaxSumExp<std::tuple<nntile::fp16_t>>;

//! Pack of maxsumexp operations for different types
maxsumexp_pack_t maxsumexp;

} // namespace nntile::haul
