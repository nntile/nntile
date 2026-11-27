/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/gelu_backward.cc
 * GeLU backward operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/gelu_backward.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>

// Other NNTile headers
#include "nntile/kernel/gelu_backward.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
GeluBackward<std::tuple<T>>::GeluBackward():
    codelet(
        "nntile_gelu_backward",
        &GeluBackward<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &GeluBackward<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &GeluBackward<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply gelu_backward on StarPU buffer on CPU
template<typename T>
void GeluBackward<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto x = ::nntile::haul::buf_chunks<T>(buffers, 0);
    auto dy = ::nntile::haul::buf_chunks<T>(buffers, 1);
    auto dx = ::nntile::haul::buf_chunks<T>(buffers, 2);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::haul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, const T *px, const T *pdy, T *pdx) {
            kernel::gelu_backward::cpu<T>(n, args->alpha, px, pdy, args->beta, pdx);
        },
        x,
        dy,
        dx);
}

// Specializations of CPU wrapper for accelerated types
template<>
void GeluBackward<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void GeluBackward<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void GeluBackward<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply gelu_backward on StarPU buffer on CUDA
template<typename T>
void GeluBackward<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto x = ::nntile::haul::buf_chunks<T>(buffers, 0);
    auto dy = ::nntile::haul::buf_chunks<T>(buffers, 1);
    auto dx = ::nntile::haul::buf_chunks<T>(buffers, 2);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::haul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, const T *px, const T *pdy, T *pdx) {
            kernel::gelu_backward::cuda<T>(
                stream,
                n,
                args->alpha,
                px,
                pdy,
                args->beta,
                pdx
            );
        },
        x,
        dy,
        dx);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void GeluBackward<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void GeluBackward<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void GeluBackward<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t GeluBackward<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    hash = ::nntile::haul::fnv1a(&args->alpha, sizeof(args->alpha), hash);
    hash = ::nntile::haul::fnv1a(&args->beta, sizeof(args->beta), hash);
    return hash;
}

template<typename T>
void GeluBackward<std::tuple<T>>::submit(int starpu_worker_hint, Index nelems, Scalar alpha,
        ::nnhaul::Handle & x, ::nnhaul::Handle & dy, Scalar beta, ::nnhaul::Handle & dx)
{
    starpu_data_access_mode dx_mode;
    if(beta == 0.0)
    {
        dx_mode = STARPU_W;
    }
    else
    {
        dx_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &x }, { STARPU_R, &dy }, { dx_mode, &dx } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class GeluBackward<std::tuple<nntile::fp64_t>>;
template class GeluBackward<std::tuple<nntile::fp32_t>>;
template class GeluBackward<std::tuple<nntile::fp32_fast_tf32_t>>;
template class GeluBackward<std::tuple<nntile::fp32_fast_fp16_t>>;
template class GeluBackward<std::tuple<nntile::fp32_fast_bf16_t>>;
template class GeluBackward<std::tuple<nntile::bf16_t>>;
template class GeluBackward<std::tuple<nntile::fp16_t>>;

//! Pack of gelu_backward operations for different types
gelu_backward_pack_t gelu_backward;

} // namespace nntile::haul
