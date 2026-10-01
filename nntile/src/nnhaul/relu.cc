/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/relu.cc
 * ReLU operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/relu.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/relu.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
Relu<std::tuple<T>>::Relu():
    codelet(
        "nntile_relu",
        &Relu<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Relu<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Relu<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::relu::cpu<T>
template<typename T>
void Relu<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::relu::cpu<T>(args->nelems, src, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void Relu<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Relu<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Relu<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Relu<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Relu<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Relu<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Define codelet pack
#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::relu::cuda<T>
template<typename T>
void Relu<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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
    kernel::relu::cuda<T>(stream, args->nelems, src, dst);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Relu<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Relu<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Relu<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Relu<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Relu<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Relu<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

template<typename T>
std::uint64_t Relu<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Relu<std::tuple<T>>::submit(int starpu_worker_hint, Index nelems, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    *args = args_t{nelems};
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Relu<std::tuple<nntile::fp64_t>>;
template class Relu<std::tuple<nntile::fp32_t>>;
template class Relu<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Relu<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Relu<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Relu<std::tuple<nntile::bf16_t>>;
template class Relu<std::tuple<nntile::fp16_t>>;

//! Pack of relu operations for different types
relu_pack_t relu;

} // namespace nntile::haul
