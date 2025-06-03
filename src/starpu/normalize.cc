/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/normalize.cc
 * Normalize operation for StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/normalize.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/normalize.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
Normalize<std::tuple<T>>::Normalize():
    codelet("nntile_normalize", footprint, cpu_funcs, cuda_funcs)
{
}

//! StarPU wrapper for kernel::normalize::cpu<T>
template<typename T>
void Normalize<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *gamma_beta = interfaces[0]->get_ptr<T>();
    const T *gamma = &gamma_beta[0], *beta = &gamma_beta[1];
    const T *sumnorm = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Launch kernel
    kernel::normalize::cpu<T>(args->m, args->n, args->k, args->l, args->eps,
            gamma, beta, sumnorm, dst);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Renormalize buffer along middle axis of StarPU buffer
template<typename T>
void Normalize<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *gamma_beta = interfaces[0]->get_ptr<T>();
    const T *gamma = &gamma_beta[0], *beta = &gamma_beta[1];
    const T *sumnorm = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::normalize::cuda<T>(stream, args->m, args->n, args->k, args->l,
            args->eps, gamma, beta, sumnorm, dst);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for normalize tasks
template<typename T>
uint32_t Normalize<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // Apply hash over parameters m, n and k. This way if we swap values of m,
    // n and k, then the total size of buffers will remain the same, but the
    // footprint will be different
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void Normalize<std::tuple<T>>::submit(Index m, Index n, Index k, Index l, Scalar eps, Handle gamma_beta,
        Handle sumnorm, Handle dst)
//! Insert normalize task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    auto args = new args_t
    {
        .m = m,
        .n = n,
        .k = k,
        .l = l,
        .eps = eps
    };
    double nflops = 14 * m * n * k;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, gamma_beta.get(),
            STARPU_R, sumnorm.get(),
            STARPU_RW, dst.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in normalize task submission");
    }
}

//! Pack of normalize operations for different types
normalize_pack_t normalize;

} // namespace nntile::starpu
