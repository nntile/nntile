/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/norm_slice.cc
 * Euclidean norms of fibers into a slice of a StarPU buffer (out-of-place version)
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/norm_slice.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/norm_slice.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked norm_slice driver (chunk_units.hh scheme; the layout
// derivation is the maxsumexp one - src1 [m, k, n] with m contiguous,
// the fiber of unit (i0, i2) taking one element per row of the panel).
// The pieces are runs of whole rows inside single src1 chunks; a piece
// computes the PARTIAL scaled norm of every unit of its panel and the
// pieces of a panel combine through the kernel's OWN hypot structure:
// the first piece runs with the caller's (alpha, beta) against the real
// src2 segment, every continuation piece runs with beta=1 and src2
// ALIASED to the accumulating dst (the kernel reads src2[i] before
// writing dst[i], element-wise, so the alias is safe on both bodies) -
// i.e. dst <- hypot(dst_partial, alpha*norm_piece), and
// hypot(hypot(a,b),c) = hypot(a,b,c) makes the chain algebraically
// exact, applied in fiber-position order on the single in-order stream.
// All-zero pieces anchor the chain because the CUDA m>1 kernels write
// the beta-combined value for all-zero fibers (0 for beta==0,
// fabs(beta*src2) otherwise), matching the CPU kernel and the CUDA m==1
// kernels - the kernel-side fix that made zero padding/masks legal
// chunked inputs. Rounding differs from the plain call by the piecewise
// hypot chain (the CUDA kernel adds its shared-memory atomicAdd
// nondeterminism, which the plain call already has).
// The dst slice segment of a panel is used in place or staged through a
// bounded m temp (zero-initialized - dst is pure output); the src2
// segment is read in place or staged through a bounded m temp; rows
// straddling src1 chunk boundaries are gathered into a bounded m temp
// and reduced as (m, 1, 1) pieces. Panel-aligned pieces may span whole
// panels. Zero new kernels; all-unchunked buffers keep the single
// classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void norm_slice_chunked(Index m, Index n, Index k, Scalar alpha,
    ::nnhaul::Buffer *buf_src1, Scalar beta, ::nnhaul::Buffer *buf_src2,
    ::nnhaul::Buffer *buf_dst, Arena &arena,
    Sub &&sub)
{
    if(m <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_src1(buf_src1);
    chunkunits::ChunkView<T> const cv_src2(buf_src2);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    chunkunits::UnitSpan<T> const span_src1{&cv_src1, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_src2{&cv_src2, std::size_t(m), k};
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t(m), k};
    bool const chunked_src1 = cv_src1.nchunks > 1;
    T *temp_src1 = chunked_src1 ? arena.alloc(std::size_t(m)) : nullptr;
    chunkunits::SliceStage<T, Arena> src2_stage(
        cv_src2, std::size_t(m), false, arena);
    chunkunits::SliceStage<T, Arena> dst_stage(
        cv_dst, std::size_t(m), true, arena);
    chunkunits::walk_unit_runs(n, k,
        [&](Index i, Index len, bool first, bool src1_str,
            bool /*src2_str*/, bool /*dst_str*/)
        {
            if(len <= 0 || !arena.ok_state())
            {
                return;
            }
            Index const panel = i / k;
            T *pd = dst_stage.advance(panel, false);
            T const *p1 = nullptr;
            if(src1_str)
            {
                chunkunits::gather_range(cv_src1, std::size_t(m) * i,
                    std::size_t(m), temp_src1, arena);
                p1 = temp_src1;
            }
            else
            {
                p1 = span_src1.ptr(i);
            }
            T const *p2 = first ? src2_stage.advance(panel, false) : pd;
            if(p1 == nullptr || p2 == nullptr || pd == nullptr
                || !arena.ok_state())
            {
                return;
            }
            Scalar const beta_eff = first ? beta : Scalar{1};
            if(len >= k)
            {
                sub(m, len / k, k, p1, beta_eff, p2, pd);
            }
            else
            {
                sub(m, Index{1}, len, p1, beta_eff, p2, pd);
            }
        },
        span_src1, span_src2, span_dst);
    dst_stage.retire();
}

} // namespace

//! Constructor
template<typename T>
NormSlice<std::tuple<T>>::NormSlice():
    codelet(
        "nntile_norm_slice",
        &NormSlice<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &NormSlice<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &NormSlice<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::norm_slice::norm_slice<T>
template<typename T>
void NormSlice<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_src1 = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_src2 = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[2]);
    if((buf_src1->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_src2->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
        && (buf_dst->chunk_count() == 1 && !chunkunits::force_chunked_kernels()))
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::norm_slice::cpu<T>(args->m, args->n, args->k, args->alpha,
            buf_src1->get_ptr<T>(), args->beta, buf_src2->get_ptr<T>(),
            buf_dst->get_ptr<T>());
        return;
    }
    Scalar const alpha = args->alpha;
    try
    {
        chunkunits::ArenaCpu<T> arena;
        norm_slice_chunked<T>(
            args->m,
            args->n,
            args->k,
            alpha,
            buf_src1,
            args->beta,
            buf_src2,
            buf_dst,
            arena,
            [alpha](Index mm, Index nn, Index kk, T const *p1, Scalar be,
                T const *p2, T *pd) {
                kernel::norm_slice::cpu<T>(mm, nn, kk, alpha, p1, be, p2, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_norm_slice: cpu chunked driver aborted after an "
            "allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void NormSlice<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void NormSlice<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void NormSlice<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! NNHaul wrapper for kernel::norm_slice::norm_slice<T>
template<typename T>
void NormSlice<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_src1 = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_src2 = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[2]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if((buf_src1->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_src2->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
        && (buf_dst->chunk_count() == 1 && !chunkunits::force_chunked_kernels()))
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::norm_slice::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, buf_src1->get_ptr<T>(), args->beta,
            buf_src2->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ArenaCuda<T> arena(stream);
    Scalar const alpha = args->alpha;
    norm_slice_chunked<T>(
        args->m,
        args->n,
        args->k,
        alpha,
        buf_src1,
        args->beta,
        buf_src2,
        buf_dst,
        arena,
        [alpha, stream](Index mm, Index nn, Index kk, T const *p1, Scalar be,
            T const *p2, T *pd) {
            kernel::norm_slice::cuda<T>(
                stream, mm, nn, kk, alpha, p1, be, p2, pd);
        });
    if(!arena.ok_state())
    {
        std::fprintf(stderr,
            "nntile_norm_slice: chunked CUDA driver aborted after a stream "
            "error; dst left unwritten\n");
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void NormSlice<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void NormSlice<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void NormSlice<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for norm_slice tasks
template<typename T>
std::uint64_t NormSlice<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
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
void NormSlice<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, Scalar alpha, ::nnhaul::Handle & src1, Scalar beta,
        ::nnhaul::Handle & src2, ::nnhaul::Handle & dst, int redux)
//! Insert norm_slice task into StarPU pool of tasks
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
    args->beta = beta;
    // Put amount of bytes read and write inplace of gflops
    size_t src_nbytes = sizeof(T) * m * k * n;
    size_t dst_nbytes = sizeof(T) * m * n;
    double nflops = beta == 0.0 ? src_nbytes + dst_nbytes :
        src_nbytes + 2*dst_nbytes;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked unit driver; norm_slice never stages.
    (void)nflops;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src1 }, { STARPU_R, &src2 }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class NormSlice<std::tuple<nntile::fp64_t>>;
template class NormSlice<std::tuple<nntile::fp32_t>>;
template class NormSlice<std::tuple<nntile::fp32_fast_tf32_t>>;
template class NormSlice<std::tuple<nntile::fp32_fast_fp16_t>>;
template class NormSlice<std::tuple<nntile::fp32_fast_bf16_t>>;
template class NormSlice<std::tuple<nntile::bf16_t>>;
template class NormSlice<std::tuple<nntile::fp16_t>>;

//! Pack of norm_slice operations for different types
norm_slice_pack_t norm_slice;

} // namespace nntile::haul
