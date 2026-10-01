/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/logsumexp.cc
 * Log of sum of exponents for StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/logsumexp.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/logsumexp.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
LogSumExp<std::tuple<T>>::LogSumExp():
    codelet(
        "nntile_logsumexp",
        &LogSumExp<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &LogSumExp<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &LogSumExp<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply logsumexp operation for StarPU buffers in CPU
template<typename T>
void LogSumExp<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *maxsumexp = ::nntile::haul::buf_as<T>(buffers, 0);
    T *logsumexp = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::logsumexp::cpu<T>(args->nelems, maxsumexp, logsumexp);
}

// Specializations of CPU wrapper for accelerated types
template<>
void LogSumExp<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void LogSumExp<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void LogSumExp<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void LogSumExp<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *maxsumexp = ::nntile::haul::buf_as<T>(buffers, 0);
    T *logsumexp = ::nntile::haul::buf_as<T>(buffers, 1);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Launch kernel
    kernel::logsumexp::cuda<T>(stream, args->nelems, maxsumexp, logsumexp);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void LogSumExp<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void LogSumExp<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void LogSumExp<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t LogSumExp<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit logsumexp task
template<typename T>
void LogSumExp<std::tuple<T>>::submit(int starpu_worker_hint,
        Index nelems, ::nnhaul::Handle & maxsumexp, ::nnhaul::Handle & logsumexp)
//! Insert logsumexp task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = new args_t();
    args->nelems = nelems;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &maxsumexp }, { STARPU_W, &logsumexp } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class LogSumExp<std::tuple<nntile::fp64_t>>;
template class LogSumExp<std::tuple<nntile::fp32_t>>;
template class LogSumExp<std::tuple<nntile::fp32_fast_tf32_t>>;
template class LogSumExp<std::tuple<nntile::fp32_fast_fp16_t>>;
template class LogSumExp<std::tuple<nntile::fp32_fast_bf16_t>>;
template class LogSumExp<std::tuple<nntile::bf16_t>>;
template class LogSumExp<std::tuple<nntile::fp16_t>>;

//! Pack of logsumexp operations for different types
logsumexp_pack_t logsumexp;

} // namespace nntile::haul
