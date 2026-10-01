/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/scale.cc
 * Scale operation on a StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/scale.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/scale.hh"
#include "nntile/nnhaul/ops/clear.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
Scale<std::tuple<T>>::Scale():
    codelet(
        "nntile_scale",
        &Scale<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Scale<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Scale<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::scale::cpu<T>
template<typename T>
void Scale<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::scale::cpu<T>(args->nelems, args->alpha, src, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void Scale<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scale<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Scale<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scale<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Scale<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scale<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::scale::cuda<T>
template<typename T>
void Scale<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Launch kernel
    kernel::scale::cuda<T>(stream, args->nelems, args->alpha, src, dst);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Scale<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scale<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Scale<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scale<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Scale<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scale<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for scale tasks that depends only on cl_arg
template<typename T>
std::uint64_t Scale<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Scale<std::tuple<T>>::submit(int starpu_worker_hint,
    Index nelems, Scalar alpha, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
{
    constexpr Scalar zero = 0.0;
    // if alpha is zero, function reduces to clear
    if(alpha == zero)
    {
        clear.submit(starpu_worker_hint, dst);
        return;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Scale<std::tuple<nntile::fp64_t>>;
template class Scale<std::tuple<nntile::fp32_t>>;
template class Scale<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Scale<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Scale<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Scale<std::tuple<nntile::bf16_t>>;
template class Scale<std::tuple<nntile::fp16_t>>;

//! Pack of scale operations for different types
scale_pack_t scale;

} // namespace nntile::haul
