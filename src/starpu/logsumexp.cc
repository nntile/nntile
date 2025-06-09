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
#include "nntile/starpu/logsumexp.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/logsumexp.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
LogSumExp<std::tuple<T>>::LogSumExp():
    codelet("nntile_logsumexp", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply logsumexp operation for StarPU buffers in CPU
template<typename T>
void LogSumExp<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *maxsumexp = interfaces[0]->get_ptr<T>();
    T *logsumexp = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::logsumexp::cpu<T>(args->nelems, maxsumexp, logsumexp);
#endif // STARPU_SIMGRID
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
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *maxsumexp = interfaces[0]->get_ptr<T>();
    T *logsumexp = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::logsumexp::cuda<T>(stream, args->nelems, maxsumexp, logsumexp);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void LogSumExp<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void LogSumExp<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void LogSumExp<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
uint32_t LogSumExp<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit logsumexp task
template<typename T>
void LogSumExp<std::tuple<T>>::submit(
        Index nelems, Handle maxsumexp, Handle logsumexp)
//! Insert logsumexp task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = new args_t();
    args->nelems = nelems;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, maxsumexp.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_W, logsumexp.get(),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in logsumexp task submission");
    }
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

//! Pack of logsumexp operations for different types
logsumexp_pack_t logsumexp;

} // namespace nntile::starpu
