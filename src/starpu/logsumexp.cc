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
#endif // NNTILE_USE_CUDA

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

//! Pack of logsumexp operations for different types
logsumexp_pack_t logsumexp;

} // namespace nntile::starpu
