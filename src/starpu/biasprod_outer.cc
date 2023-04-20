/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/biasprod_outer.cc
 * Bias-like product over outer axes operation on a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-20
 * */

#include "nntile/starpu/biasprod_outer.hh"
#include "nntile/kernel/biasprod_outer.hh"
#include <cstdlib>

namespace nntile
{
namespace starpu
{
//! StarPU wrappers for biasprod_outer operation
namespace biasprod_outer
{

//! Apply bias-like product along outer axes of StarPU buffer in CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::biasprod_outer::cpu<T>(args->m, args->n, args->k, src, dst);
}

//! Footprint for bias tasks that depends only on m, n and k
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
    codelet_fp32.init("nntile_biasprod_outer_fp32",
            footprint<fp32_t>,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_biasprod_outer_fp64",
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
void submit(Index m, Index n, Index k, Handle src, Handle dst)
//! Insert biasprod_outer task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t<T> *args = (args_t<T> *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    fp64_t nflops = m * n * k;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_RW, static_cast<starpu_data_handle_t>(dst),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in biasprod_outer task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index m, Index n, Index k, Handle src, Handle dst);

template
void submit<fp64_t>(Index m, Index n, Index k, Handle src, Handle dst);

} // namespace biasprod_outer
} // namespace starpu
} // namespace nntile

