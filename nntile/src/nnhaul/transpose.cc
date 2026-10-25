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
 * @version 1.1.0
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
#include "nntile/kernel/nnhaul/chunk_ops.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked transpose (no staging, no panel loop, no temp).
//
// Kernel index map: src is m-by-n with element (i, j) at flat i + j*m
// (i contiguous) and dst is n-by-m with element (i, j) at flat i*n + j
// (j contiguous). The chunked body calls one tiled device kernel: each
// thread computes the destination flat position from its tile
// coordinates, maps the matching source position, and both positions go
// through the chunk tables with the compile-time shift/mask map. A
// chunked buffer is therefore exactly the unchunked call plus pointer
// arithmetic. All-unchunked buffers keep the single classic call.
// ---------------------------------------------------------------------------

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
    if(buf_src->chunk_count() == 1 && buf_dst->chunk_count() == 1)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::transpose::cpu<T>(args->m, args->n, args->alpha,
            buf_src->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    // Chunked CPU body (never scheduled by this runtime): the same map
    // as the CUDA tile kernel, elementwise through the chunk views.
    using Y = typename T::repr_t;
    Y const alpha{args->alpha};
    chunkunits::ChunkView<T> const cs(buf_src, 0);
    chunkunits::ChunkView<T> const cd(buf_dst, 0);
    for(Index i = 0; i < args->m; ++i)
    {
        for(Index j = 0; j < args->n; ++j)
        {
            std::size_t const sf = std::size_t(i) + std::size_t(j) * args->m;
            std::size_t const df = std::size_t(i) * args->n + j;
            std::size_t const sc = cs.chunk_of(sf);
            std::size_t const dc = cd.chunk_of(df);
            *cd.ptr(dc, cd.chunk_off(df)) =
                static_cast<T>(alpha * Y{*cs.ptr(sc, cs.chunk_off(sf))});
        }
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
    if(buf_src->chunk_count() == 1 && buf_dst->chunk_count() == 1)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::transpose::cuda<T>(stream, args->m, args->n, args->alpha,
            buf_src->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    ::nntile::kernel::nnhaul::chunk_transpose(*buf_src, *buf_dst,
        static_cast<std::size_t>(args->m),
        static_cast<std::size_t>(args->n),
        static_cast<float>(args->alpha),
        chunkunits::nnhaul_elem_type<T>());
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
    // The body picks the unchunked classic call or the native chunked
    // tile kernel; no staging and no launch-count guard either way.
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
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
