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
 * @date 2022-09-27
 * */

#include "nntile/starpu/bias.hh"
#include "nntile/kernel/bias.hh"

namespace nntile
{
namespace starpu
{
//! StarPU wrappers for bias operation
namespace bias
{

//! Apply bias along middle axis of StarPU buffer in CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto nargc = reinterpret_cast<argc_t *>(cl_args);
    if (nargc->num_arguments == 5) {
        auto args = reinterpret_cast<args_t *>(cl_args);
        // Get interfaces
        auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
        const T *src = interfaces[0]->get_ptr<T>();
        T *dst = interfaces[1]->get_ptr<T>();
        // Launch kernel
        kernel::bias::cpu<T>(args->m, args->n, args->k, src, dst);
    } else if (nargc->num_arguments == 3) {
        auto args = reinterpret_cast<val_size_t<T> *>(cl_args);
        // Get interfaces
        auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
        T *src = interfaces[0]->get_ptr<T>();
        // Launch kernel
        kernel::bias::cpu<T>(args->val, args->nelems, src);
    }
}

#ifdef NNTILE_USE_CUDA
//! Apply bias along middle axis of StarPU buffer on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::bias::cuda<T>(stream, args->m, args->n, args->k, src, dst);
}
#endif // NNTILE_USE_CUDA

//! Footprint for bias tasks that depends only on m, n and k
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
    codelet_fp32.init("nntile_bias_fp32",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_bias_fp64",
            footprint,
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
void submit(Index m, Index n, Index k, Handle src, Handle dst)
//! Insert bias task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    // 5 is a number of argument in the called kernel function
    auto args = new args_t(5, m, n, k);
    fp64_t nflops = m * n * k;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            STARPU_CL_ARGS, args, sizeof(*args),
            Config::STARPU_RW_COMMUTE, static_cast<starpu_data_handle_t>(dst),
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
void submit<fp32_t>(Index m, Index n, Index k, Handle src, Handle dst);

template
void submit<fp64_t>(Index m, Index n, Index k, Handle src, Handle dst);


template<typename T>
void submit(T val, Index num_elements, Handle src)
{
    // Submit task
    // 3 is a number of argument in the called kernel function
    auto cl_args = new val_size_t<T>(3, val, num_elements);
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_RW, static_cast<starpu_data_handle_t>(src),
            STARPU_CL_ARGS, cl_args, sizeof(*cl_args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in bias task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(fp32_t val, Index num_elements, Handle src);

template
void submit<fp64_t>(fp64_t val, Index num_elements, Handle src);

} // namespace bias
} // namespace starpu
} // namespace nntile

