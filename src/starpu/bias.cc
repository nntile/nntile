/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/bias.cc
 * Bias operation on a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-12
 * */

#include "nntile/starpu/bias.hh"
#include "nntile/kernel/cpu/bias.hh"

#ifdef NNTILE_USE_CUDA
#   include "nntile/kernel/cuda/bias.hh"
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace starpu
{

//! Apply bias along middle axis of StarPU buffer in CPU
template<typename T>
void bias_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<bias_args *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    nntile::kernel::cpu::bias<T>(args->m, args->n, args->k, src, dst);
}

#ifdef NNTILE_USE_CUDA
//! Apply bias along middle axis of StarPU buffer on CUDA
template<typename T>
void bias_cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<bias_args *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    nntile::kernel::cuda::bias<T>(stream, args->m, args->n, args->k, src, dst);
}
#endif // NNTILE_USE_CUDA

//! Footprint for bias tasks that depends only on m, n and k
static
uint32_t bias_footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<bias_args *>(task->cl_arg);
    // Apply hash over parameters m, n and k. This way if we swap values of m,
    // n and k, then the total size of buffers will remain the same, but the
    // footprint will be different
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    return hash;
}

StarpuCodelet bias_codelet_fp32("nntile_bias_fp32",
        bias_footprint,
        {bias_cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
        {bias_cuda<fp32_t>}
#else // NNTILE_USE_CUDA
        {}
#endif // NNTILE_USE_CUDA
        );

StarpuCodelet bias_codelet_fp64("nntile_bias_fp64",
        bias_footprint,
        {bias_cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
        {bias_cuda<fp64_t>}
#else // NNTILE_USE_CUDA
        {}
#endif // NNTILE_USE_CUDA
        );

void bias_restrict_where(uint32_t where)
{
    bias_codelet_fp32.restrict_where(where);
    bias_codelet_fp64.restrict_where(where);
}

void bias_restore_where()
{
    bias_codelet_fp32.restore_where();
    bias_codelet_fp64.restore_where();
}

template<typename T>
constexpr StarpuCodelet *bias_codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr StarpuCodelet *bias_codelet<fp32_t>()
{
    return &bias_codelet_fp32;
}

template<>
constexpr StarpuCodelet *bias_codelet<fp64_t>()
{
    return &bias_codelet_fp64;
}

template<typename T>
void bias(Index m, Index n, Index k, starpu_data_handle_t src,
        starpu_data_handle_t dst)
//! Insert bias task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    auto args = new bias_args
    {
        .m = m,
        .n = n,
        .k = k
    };
    fp64_t nflops = m * n * k;
    // Submit task
    int ret = starpu_task_insert(bias_codelet<T>(),
            STARPU_R, src,
            STARPU_CL_ARGS, args, sizeof(*args),
            Starpu::STARPU_RW_COMMUTE, dst,
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in bias task submission");
    }
}

// Explicit instantiation
template
void bias<fp32_t>(Index m, Index n, Index k, starpu_data_handle_t src,
        starpu_data_handle_t dst);

template
void bias<fp64_t>(Index m, Index n, Index k, starpu_data_handle_t src,
        starpu_data_handle_t dst);

} // namespace starpu
} // namespace nntile

