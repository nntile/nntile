/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/silu_backward.cc
 * Backward SiLU operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/silu_backward.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/silu_backward.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
SiluBackward<std::tuple<T>>::SiluBackward():
    codelet(
        "nntile_silu_backward",
        &SiluBackward<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &SiluBackward<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &SiluBackward<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::silu_backward::cpu<T>
template<typename T>
void SiluBackward<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *x = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *dy = ::nntile::haul::buf_as<T>(buffers, 1);
    T *dx = ::nntile::haul::buf_as<T>(buffers, 2);
    // Launch kernel
    kernel::silu_backward::cpu<T>(args->nelems, args->alpha, x, dy, args->beta, dx);
}

// Specializations of CPU wrapper for accelerated types
template<>
void SiluBackward<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SiluBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SiluBackward<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SiluBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SiluBackward<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SiluBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::silu_backward::cuda<T>
template<typename T>
void SiluBackward<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *x = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *dy = ::nntile::haul::buf_as<T>(buffers, 1);
    T *dx = ::nntile::haul::buf_as<T>(buffers, 2);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Launch kernel
    kernel::silu_backward::cuda<T>(
        stream,
        args->nelems,
        args->alpha,
        x,
        dy,
        args->beta,
        dx
    );
}

// Specializations of CUDA wrapper for accelerated types
template<>
void SiluBackward<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SiluBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SiluBackward<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SiluBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SiluBackward<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SiluBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for silu_backward tasks
template<typename T>
std::uint64_t SiluBackward<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    hash = ::nntile::haul::fnv1a(&args->alpha, sizeof(args->alpha), hash);
    hash = ::nntile::haul::fnv1a(&args->beta, sizeof(args->beta), hash);
    return hash;
}

template<typename T>
void SiluBackward<std::tuple<T>>::submit(int starpu_worker_hint, Index nelems, Scalar alpha,
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
template class SiluBackward<std::tuple<nntile::fp64_t>>;
template class SiluBackward<std::tuple<nntile::fp32_t>>;
template class SiluBackward<std::tuple<nntile::fp32_fast_tf32_t>>;
template class SiluBackward<std::tuple<nntile::fp32_fast_fp16_t>>;
template class SiluBackward<std::tuple<nntile::fp32_fast_bf16_t>>;
template class SiluBackward<std::tuple<nntile::fp16_t>>;
template class SiluBackward<std::tuple<nntile::bf16_t>>;

//! Pack of silu_backward operations for different types
silu_backward_pack_t silu_backward;

} // namespace nntile::haul
