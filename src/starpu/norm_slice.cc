/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/norm_slice.cc
 * Euclidean norms of fibers into a slice of a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-19
 * */

#include "nntile/starpu/norm_slice.hh"
#include "nntile/kernel/norm_slice.hh"
#include <cstdlib>

namespace nntile
{
namespace starpu
{
//! StarPU wrappers for norm_slice operation
namespace norm_slice
{

//! StarPU wrapper for kernel::norm_slice::cpu<T>
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
    kernel::norm_slice::cpu<T>(args->m, args->n, args->k, args->alpha, src,
            args->beta, dst);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::norm_slice::cuda<T>
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::norm_slice::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, src, args->beta, dst);
}
#endif // NNTILE_USE_CUDA

//! Footprint for norm_slice tasks
template<typename T>
static
uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(task->cl_arg);
    // Apply hash over parameters m, n and k
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    return hash;
}

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_norm_slice_fp32",
            footprint<fp32_t>,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_norm_slice_fp64",
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
void submit(Index m, Index n, Index k, T alpha, Handle src, T beta,
        Handle dst, int redux)
//! Insert norm_slice task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Access mode for the dst handle
    constexpr T zero = 0, one = 1;
    enum starpu_data_access_mode dst_mode;
    if(beta == zero)
    {
        dst_mode = STARPU_W;
    }
    else if(beta == one)
    {
        if(redux != 0)
        {
            //dst_mode = STARPU_REDUX;
            dst_mode = Config::STARPU_RW_COMMUTE;
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
            dst_mode, static_cast<starpu_data_handle_t>(dst),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in norm_slice task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index m, Index n, Index k, fp32_t alpha, Handle src,
        fp32_t beta, Handle dst, int redux);

template
void submit<fp64_t>(Index m, Index n, Index k, fp64_t alpha, Handle src,
        fp64_t beta, Handle dst, int redux);

} // namespace norm_slice
} // namespace starpu
} // namespace nntile

