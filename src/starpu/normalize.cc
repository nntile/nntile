/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/normalize.cc
 * Normalize operation for StarPU buffer
 *
 * @version 1.1.0
 * */

#ifndef STARPU_SIMGRID
#include "nntile/kernel/normalize.hh"
#endif // STARPU_SIMGRID
#include "nntile/starpu/normalize.hh"

namespace nntile::starpu::normalize
{

//! Renormalize buffer along middle axis of StarPU buffer
template<typename T>
void cpu(void *buffers[], void *cl_args)
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
void cuda(void *buffers[], void *cl_args)
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

//! Footprint for normalize tasks that depends only on m, n and k
template<typename T>
static
uint32_t footprint(struct starpu_task *task)
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

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_normalize_fp32",
            footprint<fp32_t>,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_normalize_fp64",
            footprint<fp64_t>,
            {cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp64_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
}

void restrict_where(uint32_t where)
{
    codelet_fp32.restrict_where(where);
    codelet_fp64.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_fp64.restore_where();
}

template<typename T>
void submit(Index m, Index n, Index k, Index l, Scalar eps, Handle gamma_beta,
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
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(gamma_beta),
            STARPU_R, static_cast<starpu_data_handle_t>(sumnorm),
            STARPU_RW, static_cast<starpu_data_handle_t>(dst),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in normalize task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index m, Index n, Index k, Index l, Scalar eps,
        Handle gamma_beta, Handle sumnorm, Handle dst);

template
void submit<fp64_t>(Index m, Index n, Index k, Index l, Scalar eps,
        Handle gamma_beta, Handle sumnorm, Handle dst);

} // namespace nntile::starpu::normalize
