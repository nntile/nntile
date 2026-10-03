/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/fill.cc
 * Fill operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/fill.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/fill.hh"
#ifdef NNTILE_USE_CUDA
#include <cuda_runtime.h>
#endif // NNTILE_USE_CUDA

namespace nntile::haul
{

//! Constructor
template<typename T>
Fill<std::tuple<T>>::Fill():
    codelet(
        "nntile_fill",
        &Fill<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Fill<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Fill<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply fill on StarPU buffer on CPU
template<typename T>
void Fill<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    T *data = ::nntile::haul::buf_as<T>(buffers, 0);
    // Launch kernel
    kernel::fill::cpu<T>(args->nelems, args->value, data);
}

// Specializations of CPU wrapper for accelerated types
template<>
void Fill<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Fill<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Fill<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Fill<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Fill<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Fill<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
namespace
{

struct NotedPtr
{
    int type = -1;
    int device = -1;
    void const *ptr = nullptr;
    bool on_device = false;
};

// Flushed before the query and again with the attribute numbers.
// A fault in either still leaves nelems and the pointer in the ctest log.
NotedPtr note_device_ptr(
    char const *codelet, long long nelems, void const *ptr, int stream_device)
{
    std::fprintf(stderr, "%s nelems=%lld ptr=%p\n", codelet, nelems, ptr);
    std::fflush(stderr);
    cudaPointerAttributes attr{};
    cudaError_t const query = cudaPointerGetAttributes(&attr, ptr);
    NotedPtr noted;
    noted.ptr = ptr;
    if (query != cudaSuccess)
    {
        cudaGetLastError();
    }
    else
    {
        noted.type = static_cast<int>(attr.type);
        noted.device = attr.device;
        noted.on_device =
            noted.type == static_cast<int>(cudaMemoryTypeDevice) &&
            noted.device == stream_device;
    }
    std::fprintf(stderr, "%s nelems=%lld type=%d device=%d ptr=%p\n",
        codelet, nelems, noted.type, noted.device, ptr);
    std::fflush(stderr);
    return noted;
}

void require_device_ptr(
    char const *codelet, long long nelems, NotedPtr const &noted)
{
    if (nelems < 1 || !noted.on_device)
    {
        char text[192];
        std::snprintf(text, sizeof(text),
            "%s nelems=%lld type=%d device=%d ptr=%p",
            codelet, nelems, noted.type, noted.device, noted.ptr);
        throw std::runtime_error(text);
    }
}

} // namespace

//! Apply fill on StarPU buffer on CUDA
template<typename T>
void Fill<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    T *data = ::nntile::haul::buf_as<T>(buffers, 0);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    int const stream_device = ::nnhaul::device_index();
    long long const nelems = static_cast<long long>(args->nelems);
    NotedPtr const noted =
        note_device_ptr("nntile_fill", nelems, data, stream_device);
    require_device_ptr("nntile_fill", nelems, noted);
    // Launch kernel
    kernel::fill::cuda<T>(stream, args->nelems, args->value, data);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Fill<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Fill<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Fill<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Fill<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Fill<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Fill<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t Fill<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Fill<std::tuple<T>>::submit(int starpu_worker_hint, Index nelems, Scalar value, ::nnhaul::Handle & data)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->value = value;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_W, &data } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Fill<std::tuple<nntile::fp64_t>>;
template class Fill<std::tuple<nntile::fp32_t>>;
template class Fill<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Fill<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Fill<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Fill<std::tuple<nntile::bf16_t>>;
template class Fill<std::tuple<nntile::fp16_t>>;

//! Pack of fill operations for different types
fill_pack_t fill;

} // namespace nntile::haul
