/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/sumprod_fiber.cc
 * Sums over slices into a fiber of a product of two StarPU buffers
 *
 * @version 1.1.0
 * */

#ifndef STARPU_SIMGRID
#include "nntile/kernel/sumprod_fiber.hh"
#endif // STARPU_SIMGRID
#include "nntile/starpu/sumprod_fiber.hh"
#include <cstdlib>

namespace nntile::starpu::sumprod_fiber
{

//! StarPU wrapper for kernel::sumprod_fiber::cpu<T>
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src1 = interfaces[0]->get_ptr<T>();
    const T *src2 = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Launch kernel
    kernel::sumprod_fiber::cpu<T>(args->m, args->n, args->k, args->alpha,
            src1, src2, args->beta, dst);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::sumprod_fiber::cuda<T>
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src1 = interfaces[0]->get_ptr<T>();
    const T *src2 = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::sumprod_fiber::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, src1, src2, args->beta, dst);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for sumprod_fiber tasks
static
uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // Hash over alpha
    uint32_t hash = args->alpha == Scalar{0} ? -1 : 0;
    // Apply hash over parameters m, n and k
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    return hash;
}

Codelet codelet_fp32, codelet_fp64, codelet_fp32_fast_tf32, codelet_bf16;

void init()
{
    codelet_fp32.init("nntile_sumprod_fiber_fp32",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

        codelet_bf16.init("nntile_sumprod_fiber_bf16",
            footprint,
            {cpu<bf16_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<bf16_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_tf32.init("nntile_sumprod_fiber_fp32_fast_tf32",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp64.init("nntile_sumprod_fiber_fp64",
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
void submit(Index m, Index n, Index k, Scalar alpha, Handle src1, Handle src2,
        Scalar beta, Handle dst, int redux)
//! Insert sumprod_fiber task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Access mode for the dst handle
    constexpr Scalar zero = 0, one = 1;
    enum starpu_data_access_mode dst_mode;
    if(beta == zero)
    {
        dst_mode = STARPU_W;
    }
    else if(beta == one)
    {
        if(redux != 0)
        {
            dst_mode = STARPU_REDUX;
            //dst_mode = Config::STARPU_RW_COMMUTE;
        }
        else
        {
            dst_mode = Config::STARPU_RW_COMMUTE;
        }
    }
    else
    {
        dst_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->alpha = alpha;
    args->beta = beta;
    double nflops = k * (2*m*n);
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
        STARPU_R, static_cast<starpu_data_handle_t>(src1),
        STARPU_R, static_cast<starpu_data_handle_t>(src2),
        STARPU_CL_ARGS, args, sizeof(*args),
        dst_mode, static_cast<starpu_data_handle_t>(dst),
        STARPU_FLOPS, nflops,
        0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in sumprod_fiber task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index m, Index n, Index k, Scalar alpha, Handle src1,
        Handle src2, Scalar beta, Handle dst, int redux);

template
void submit<bf16_t>(Index m, Index n, Index k, Scalar alpha, Handle src1,
        Handle src2, Scalar beta, Handle dst, int redux);

template
void submit<fp32_fast_tf32_t>(Index m, Index n, Index k, Scalar alpha, Handle src1,
        Handle src2, Scalar beta, Handle dst, int redux);

template
void submit<fp64_t>(Index m, Index n, Index k, Scalar alpha, Handle src1,
        Handle src2, Scalar beta, Handle dst, int redux);

} // namespace nntile::starpu::sumprod_fiber
