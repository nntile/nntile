/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/sum.cc
 * Sum of slices of a StarPU buffer (outer version)
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-24
 * */

#include "nntile/starpu/sum_outer.hh"
#include "nntile/kernel/sum_outer.hh"
#include <cstdlib>

namespace nntile
{
namespace starpu
{
namespace sum_outer
{

//! Sum along outer axes of StarPU buffer on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *sum_dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::sum_outer::cpu<T>(args->m, args->n, args->k, args->alpha, src,
            args->beta, sum_dst);
}

//! Footprint for sum tasks that depends only on m, n and k
template<typename T>
static
uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(task->cl_arg);
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
    codelet_fp32.init("nntile_sum_outer_fp32",
            footprint<fp32_t>,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_sum_outer_fp64",
            footprint<fp64_t>,
            {cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
            {}
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
void submit(Index m, Index n, Index k, T alpha, Handle src, T beta,
        Handle sum_dst)
//! Insert sum_outer task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Access mode for the sum_dst handle
    constexpr T zero = 0, one = 1;
    enum starpu_data_access_mode sum_dst_mode;
    if(beta == zero)
    {
        sum_dst_mode = STARPU_W;
    }
    else if(beta == one)
    {
        sum_dst_mode = Config::STARPU_RW_COMMUTE;
    }
    else
    {
        sum_dst_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t<T> *args = (args_t<T> *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->alpha = alpha;
    args->beta = beta;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            STARPU_CL_ARGS, args, sizeof(*args),
            sum_dst_mode, static_cast<starpu_data_handle_t>(sum_dst),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in sum_outer task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index m, Index n, Index k, fp32_t alpha, Handle src,
        fp32_t beta, Handle sum_dst);

template
void submit<fp64_t>(Index m, Index n, Index k, fp64_t alpha, Handle src,
        fp64_t beta, Handle sum_dst);

} // namespace sum_outer
} // namespace starpu
} // namespace nntile

