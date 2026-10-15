/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/sumprod_slice.cc
 * Sums over fibers into a slice of a product of two StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/sumprod_slice.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/sumprod_slice.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked sumprod_slice driver (chunk_units.hh scheme; the layout
// derivation is the maxsumexp/sum_slice one - both sources [m, k, n]
// with m contiguous). The pieces are runs of whole rows inside single
// chunks of BOTH sources; a piece computes the PARTIAL Kahan
// sum-of-products of every unit of its panel and the pieces of a panel
// chain through the kernel's OWN beta!=0 accumulate contract: first
// piece (alpha, caller beta), continuations (alpha, 1) - algebraically
// exact, applied in fiber-position order on the single in-order stream,
// equal to the plain call up to the rounding of the piecewise merge (the
// CUDA kernel adds its shared-memory atomicAdd nondeterminism, which the
// plain call already has). The dst slice segment of a panel is used in
// place or staged through a bounded m temp for the panel's chain
// (gathered when the caller's beta != 0, zero-initialized otherwise,
// scattered back at the panel switch); rows straddling a src1/src2 chunk
// boundary are gathered into bounded m temps and reduced as (m, 1, 1)
// pieces. Panel-aligned pieces may span whole panels. Zero new kernels;
// all-unchunked buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void sumprod_slice_chunked(Index m, Index n, Index k, Scalar alpha,
    ::nnhaul::Buffer *buf_src1, ::nnhaul::Buffer *buf_src2, Scalar beta,
    ::nnhaul::Buffer *buf_dst, std::size_t force_bytes, Arena &arena,
    Sub &&sub)
{
    if(m <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_src1(buf_src1, force_bytes);
    chunkunits::ChunkView<T> const cv_src2(buf_src2, force_bytes);
    chunkunits::ChunkView<T> const cv_dst(buf_dst, force_bytes);
    chunkunits::UnitSpan<T> const span_src1{&cv_src1, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_src2{&cv_src2, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t(m), k};
    bool const chunked_src1 = cv_src1.nchunks > 1 || force_bytes != 0;
    bool const chunked_src2 = cv_src2.nchunks > 1 || force_bytes != 0;
    T *temp_src1 = chunked_src1 ? arena.alloc(std::size_t(m)) : nullptr;
    T *temp_src2 = chunked_src2 ? arena.alloc(std::size_t(m)) : nullptr;
    chunkunits::SliceStage<T, Arena> dst_stage(
        cv_dst, std::size_t(m), true, arena);
    chunkunits::walk_unit_runs(n, k,
        [&](Index i, Index len, bool first, bool s1_str, bool s2_str,
            bool /*dst_str*/)
        {
            if(len <= 0 || !arena.ok_state())
            {
                return;
            }
            Index const panel = i / k;
            T *pd = dst_stage.advance(panel, first && beta != Scalar{0});
            T const *p1 = nullptr;
            if(s1_str)
            {
                chunkunits::gather_range(cv_src1, std::size_t(m) * i,
                    std::size_t(m), temp_src1, arena);
                p1 = temp_src1;
            }
            else
            {
                p1 = span_src1.ptr(i);
            }
            T const *p2 = nullptr;
            if(s2_str)
            {
                chunkunits::gather_range(cv_src2, std::size_t(m) * i,
                    std::size_t(m), temp_src2, arena);
                p2 = temp_src2;
            }
            else
            {
                p2 = span_src2.ptr(i);
            }
            if(p1 == nullptr || p2 == nullptr || pd == nullptr
                || !arena.ok_state())
            {
                return;
            }
            Scalar const beta_eff = first ? beta : Scalar{1};
            if(len >= k)
            {
                sub(m, len / k, k, p1, p2, beta_eff, pd);
            }
            else
            {
                sub(m, Index{1}, len, p1, p2, beta_eff, pd);
            }
        },
        span_src1, span_src2, span_dst);
    dst_stage.retire();
}

} // namespace

//! Constructor
template<typename T>
SumProdSlice<std::tuple<T>>::SumProdSlice():
    codelet(
        "nntile_sumprod_slice",
        &SumProdSlice<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &SumProdSlice<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &SumProdSlice<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::sumprod_slice::cpu<T>
template<typename T>
void SumProdSlice<std::tuple<T>>::cpu(void *buffers[], void *cl_args) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_src1 = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_src2 = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[2]);
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    if(buf_src1->chunk_count() == 1 && buf_src2->chunk_count() == 1
        && buf_dst->chunk_count() == 1 && forced == 0)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::sumprod_slice::cpu<T>(args->m, args->n, args->k, args->alpha,
            buf_src1->get_ptr<T>(), buf_src2->get_ptr<T>(), args->beta,
            buf_dst->get_ptr<T>());
        return;
    }
    Scalar const alpha = args->alpha;
    try
    {
        chunkunits::ArenaCpu<T> arena;
        sumprod_slice_chunked<T>(
            args->m,
            args->n,
            args->k,
            alpha,
            buf_src1,
            buf_src2,
            args->beta,
            buf_dst,
            forced,
            arena,
            [alpha](Index mm, Index nn, Index kk, T const *p1, T const *p2,
                Scalar be, T *pd) {
                kernel::sumprod_slice::cpu<T>(mm, nn, kk, alpha, p1, p2, be,
                    pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_sumprod_slice: cpu chunked driver aborted after an "
            "allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void SumProdSlice<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SumProdSlice<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SumProdSlice<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! NNHaul wrapper for kernel::sumprod_slice::cuda<T>
template<typename T>
void SumProdSlice<std::tuple<T>>::cuda(void *buffers[], void *cl_args) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_src1 = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_src2 = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[2]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    if(buf_src1->chunk_count() == 1 && buf_src2->chunk_count() == 1
        && buf_dst->chunk_count() == 1 && forced == 0)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::sumprod_slice::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, buf_src1->get_ptr<T>(), buf_src2->get_ptr<T>(),
            args->beta, buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ArenaCuda<T> arena(stream);
    Scalar const alpha = args->alpha;
    sumprod_slice_chunked<T>(
        args->m,
        args->n,
        args->k,
        alpha,
        buf_src1,
        buf_src2,
        args->beta,
        buf_dst,
        forced,
        arena,
        [alpha, stream](Index mm, Index nn, Index kk, T const *p1,
            T const *p2, Scalar be, T *pd) {
            kernel::sumprod_slice::cuda<T>(
                stream, mm, nn, kk, alpha, p1, p2, be, pd);
        });
    if(!arena.ok_state())
    {
        std::fprintf(stderr,
            "nntile_sumprod_slice: chunked CUDA driver aborted after a "
            "stream error; dst left unwritten\n");
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void SumProdSlice<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SumProdSlice<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SumProdSlice<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for sumprod_slice tasks
template<typename T>
std::uint64_t SumProdSlice<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
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
void SumProdSlice<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, Scalar alpha, ::nnhaul::Handle & src1, ::nnhaul::Handle & src2,
        Scalar beta, ::nnhaul::Handle & dst, int redux)
//! Insert sumprod_slice task into StarPU pool of tasks
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
    size_t src1_nbytes = sizeof(T) * m * k * n;
    size_t dst_nbytes = sizeof(T) * m * n;
    double nflops = beta == 0.0 ? 2*src1_nbytes + dst_nbytes :
        2 * (src1_nbytes+dst_nbytes);
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked unit driver; sumprod_slice never stages.
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src1 }, { STARPU_R, &src2 }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class SumProdSlice<std::tuple<nntile::fp64_t>>;
template class SumProdSlice<std::tuple<nntile::fp32_t>>;
template class SumProdSlice<std::tuple<nntile::fp32_fast_tf32_t>>;
template class SumProdSlice<std::tuple<nntile::fp32_fast_fp16_t>>;
template class SumProdSlice<std::tuple<nntile::fp32_fast_bf16_t>>;
template class SumProdSlice<std::tuple<nntile::bf16_t>>;
template class SumProdSlice<std::tuple<nntile::fp16_t>>;

//! Pack of sumprod_slice operations for different types
sumprod_slice_pack_t sumprod_slice;

} // namespace nntile::haul
