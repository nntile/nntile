/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/add.cc
 * Add operation on a StarPU buffers
 *
 * @version 1.1.0
 * */

#ifndef STARPU_SIMGRID
#include "nntile/kernel/add.hh"
#endif // STARPU_SIMGRID
#include "nntile/starpu/add.hh"
#include "nntile/starpu/scal.hh"
#include "nntile/starpu/clear.hh"
#include "nntile/starpu/scal_inplace.hh"
#include <cstdlib>

//! StarPU wrappers for add operation
namespace nntile::starpu::add
{

//! Apply add operation for StarPU buffers in CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::add::cpu<T>(args->nelems, args->alpha, src, args->beta, dst);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply add for StarPU buffers on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::add::cuda<T>(stream, args->nelems, args->alpha, src,
            args->beta, dst);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
static
uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // Apply hash over parameters m, n, k and k_size.
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

Codelet codelet_fp32, codelet_fp64, codelet_fp32_fast_tf32, codelet_bf16;

void init()
{
    codelet_fp32.init("nntile_add_fp32",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_bf16.init("nntile_add_bf16",
            footprint,
            {cpu<bf16_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<bf16_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_tf32.init("nntile_add_fp32_fast_tf32",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp64.init("nntile_add_fp64",
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
    codelet_bf16.restrict_where(where);
    codelet_fp32_fast_tf32.restrict_where(where);
    codelet_fp64.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_bf16.restore_where();
    codelet_fp32_fast_tf32.restore_where();
    codelet_fp64.restore_where();
}

template<typename T>
void submit(Index nelems, Scalar alpha, Handle src, Scalar beta, Handle dst)
//! Insert add task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    constexpr Scalar zero = 0, one = 1;
    // If beta is zero this function reduces to scal
    if(beta == zero)
    {
        scal::submit<T>(nelems, alpha, src, dst);
        return;
    }
    // If beta is non-zero and alpha is zero then reduce to scal_inplace
    if(alpha == zero)
    {
        scal_inplace::submit<T>(nelems, beta, dst);
        return;
    }
    // Access mode for the dst handle
    enum starpu_data_access_mode dst_mode;
    if(beta == one)
    {
        dst_mode = Config::STARPU_RW_COMMUTE;
    }
    else
    {
        dst_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            STARPU_CL_ARGS, args, sizeof(*args),
            dst_mode, static_cast<starpu_data_handle_t>(dst),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in add task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index nelems, Scalar alpha, Handle src, Scalar beta,
        Handle dst);

template
void submit<bf16_t>(Index nelems, Scalar alpha, Handle src, Scalar beta,
        Handle dst);

template
void submit<fp32_fast_tf32_t>(Index nelems, Scalar alpha, Handle src, Scalar beta,
        Handle dst);

template
void submit<fp64_t>(Index nelems, Scalar alpha, Handle src, Scalar beta,
        Handle dst);

} // namespace nntile::starpu::add
