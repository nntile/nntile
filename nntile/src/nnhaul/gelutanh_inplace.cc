/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/gelutanh_inplace.cc
 * Approximate GeLU operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/gelutanh_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/gelutanh_inplace.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
GeluTanhInplace<std::tuple<T>>::GeluTanhInplace():
    codelet(
        "nntile_gelutanh_inplace",
        &GeluTanhInplace<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &GeluTanhInplace<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &GeluTanhInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply approximate gelu on StarPU buffer on CPU
template<typename T>
void GeluTanhInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    T *data = ::nntile::haul::buf_as<T>(buffers, 0);
    // Launch kernel
    kernel::gelutanh_inplace::cpu<T>(args->nelems, data);
}

// Specializations of CPU wrapper for accelerated types
template<>
void GeluTanhInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanhInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void GeluTanhInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanhInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void GeluTanhInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanhInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply approximate gelu on StarPU buffer on CUDA
template<typename T>
void GeluTanhInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    T *data = ::nntile::haul::buf_as<T>(buffers, 0);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Launch kernel
    kernel::gelutanh_inplace::cuda<T>(stream, args->nelems, data);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void GeluTanhInplace<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanhInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void GeluTanhInplace<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanhInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void GeluTanhInplace<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanhInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t GeluTanhInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void GeluTanhInplace<std::tuple<T>>::submit(int starpu_worker_hint, Index nelems, ::nnhaul::Handle & data)
{
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_RW, &data } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class GeluTanhInplace<std::tuple<nntile::fp64_t>>;
template class GeluTanhInplace<std::tuple<nntile::fp32_t>>;
template class GeluTanhInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class GeluTanhInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class GeluTanhInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class GeluTanhInplace<std::tuple<nntile::bf16_t>>;

//! Pack of gelutanh_inplace operations for different types
gelutanh_inplace_pack_t gelutanh_inplace;

} // namespace nntile::haul
