/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/softmax_inplace.cc
 * Inplace softmax operation for StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/softmax_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/softmax_inplace.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked softmax_inplace driver (chunk_units.hh scheme; see the
// index-map derivation in softmax.cc - the op is ELEMENTWISE given the
// maxsumexp pairs, so whole-row pieces reproduce the plain call's
// per-element operations bitwise). The dst tile is read-write: a row
// straddling a dst chunk boundary is gathered into a bounded m-element
// temp, transformed in place as (m, 1, 1) and scattered back; the
// maxsumexp read of a piece is its panel's 2*m segment, staged through a
// bounded temp when it straddles. Zero new kernels; all-unchunked
// buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void softmax_inplace_chunked(Index m, Index n, Index k,
    ::nnhaul::Buffer *buf_mse, ::nnhaul::Buffer *buf_dst,
    std::size_t force_bytes, Arena &arena, Sub &&sub)
{
    if(m <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_mse(buf_mse, force_bytes);
    chunkunits::ChunkView<T> const cv_dst(buf_dst, force_bytes);
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_mse{&cv_mse, std::size_t{2} * m, k};
    bool const chunked_dst = cv_dst.nchunks > 1 || force_bytes != 0;
    T *temp_dst = chunked_dst ? arena.alloc(std::size_t(m)) : nullptr;
    chunkunits::SliceStage<T, Arena> mse_stage(
        cv_mse, std::size_t{2} * m, false, arena);
    chunkunits::walk_unit_runs(n, k,
        [&](Index i, Index len, bool /*first*/, bool dst_str,
            bool /*mse_str*/)
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
            T *pm = mse_stage.advance(i / k, false);
            if(pd == nullptr || pm == nullptr || !arena.ok_state())
            {
                return;
            }
            if(len >= k)
            {
                sub(m, len / k, k, pm, pd);
            }
            else
            {
                sub(m, Index{1}, len, pm, pd);
            }
            if(dst_str)
            {
                chunkunits::scatter_range(cv_dst, std::size_t(m) * i,
                    std::size_t(m), temp_dst, arena);
            }
        },
        span_dst, span_mse);
}

} // namespace

//! Constructor
template<typename T>
SoftmaxInplace<std::tuple<T>>::SoftmaxInplace():
    codelet(
        "nntile_softmax_inplace",
        &SoftmaxInplace<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &SoftmaxInplace<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &SoftmaxInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::softmax_inplace::cpu<T>
template<typename T>
void SoftmaxInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_mse = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    if(buf_mse->chunk_count() == 1 && buf_dst->chunk_count() == 1
        && forced == 0)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::softmax_inplace::cpu<T>(args->m, args->n, args->k,
            buf_mse->get_ptr<T>(), args->alpha, buf_dst->get_ptr<T>());
        return;
    }
    Scalar const alpha = args->alpha;
    try
    {
        chunkunits::ArenaCpu<T> arena;
        softmax_inplace_chunked<T>(
            args->m,
            args->n,
            args->k,
            buf_mse,
            buf_dst,
            forced,
            arena,
            [alpha](Index mm, Index nn, Index kk, T const *pm, T *pd) {
                kernel::softmax_inplace::cpu<T>(mm, nn, kk, pm, alpha, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_softmax_inplace: cpu chunked driver aborted after an "
            "allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void SoftmaxInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SoftmaxInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SoftmaxInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SoftmaxInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SoftmaxInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SoftmaxInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! NNHaul wrapper for kernel::softmax_inplace::cuda<T>
template<typename T>
void SoftmaxInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_mse = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    if(buf_mse->chunk_count() == 1 && buf_dst->chunk_count() == 1
        && forced == 0)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::softmax_inplace::cuda<T>(stream, args->m, args->n, args->k,
            buf_mse->get_ptr<T>(), args->alpha, buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ArenaCuda<T> arena(stream);
    Scalar const alpha = args->alpha;
    softmax_inplace_chunked<T>(
        args->m,
        args->n,
        args->k,
        buf_mse,
        buf_dst,
        forced,
        arena,
        [alpha, stream](Index mm, Index nn, Index kk, T const *pm, T *pd) {
            kernel::softmax_inplace::cuda<T>(
                stream, mm, nn, kk, pm, alpha, pd);
        });
    if(!arena.ok_state())
    {
        std::fprintf(stderr,
            "nntile_softmax_inplace: chunked CUDA driver aborted after a "
            "stream error; dst left unwritten\n");
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void SoftmaxInplace<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SoftmaxInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SoftmaxInplace<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SoftmaxInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SoftmaxInplace<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SoftmaxInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for softmax_inplace tasks that depends only on m, n and k
template<typename T>
std::uint64_t SoftmaxInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, n and k. This way if we swap values of m,
    // n and k, then the total size of buffers will remain the same, but the
    // footprint will be different
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void SoftmaxInplace<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, ::nnhaul::Handle & maxsumexp, Scalar alpha,
        ::nnhaul::Handle & dst)
//! Insert softmax_inplace task into StarPU pool of tasks
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
    // chunked unit driver; softmax_inplace never stages.
    (void)nflops;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &maxsumexp }, { STARPU_RW, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class SoftmaxInplace<std::tuple<nntile::fp64_t>>;
template class SoftmaxInplace<std::tuple<nntile::fp32_t>>;
template class SoftmaxInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class SoftmaxInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class SoftmaxInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class SoftmaxInplace<std::tuple<nntile::bf16_t>>;
template class SoftmaxInplace<std::tuple<nntile::fp16_t>>;

//! Pack of softmax_inplace operations for different types
softmax_inplace_pack_t softmax_inplace;

} // namespace nntile::haul
