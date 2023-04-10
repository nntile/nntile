/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/scalprod.cc
 * Scalar product of slices for two StarPU buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-03-26
 * */

#include "nntile/starpu/scalprod.hh"
#include "nntile/kernel/scalprod.hh"
#include <cstdlib>

namespace nntile
{
namespace starpu
{
namespace scalprod
{

//! Scalar products along middle axis of StarPU buffers on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src1 = interfaces[0]->get_ptr<T>();
    const T *src2 = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Launch kernel
    kernel::scalprod::cpu<T>(args->m, args->n, args->k, args->alpha, src1,
            src2, args->beta, dst);
}

#ifdef NNTILE_USE_CUDA
//! Scalar products along middle axis of StarPU buffers on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src1 = interfaces[0]->get_ptr<T>();
    const T *src2 = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::scalprod::cuda<T>(stream, args->m, args->n, args->k, args->alpha,
            src1, src2, args->beta, dst);
}
#endif // NNTILE_USE_CUDA

//! Footprint for SCALPROD tasks that depends only on M, N, K and alpha
template<typename T>
static
uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(task->cl_arg);
    // In case alpha is zero, entire scalprod is unnecessary so it is better to
    // give it a different footprint since scalprod time will be totally
    // different
    uint32_t hash = args->alpha == T{0} ? -1 : 0;
    // Apply hash over parameters M, N and K. This way if we swap values of M,
    // N and K total size of buffers will remain the same, but the footprint
    // will be different
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    return hash;
}

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_scalprod_fp32",
            footprint<fp32_t>,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_scalprod_fp64",
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
void submit(Index m, Index n, Index k, T alpha, Handle src1, Handle src2,
        T beta, Handle dst)
//! Insert scalprod task into StarPU pool of tasks
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
    args->alpha = alpha;
    args->beta = beta;
    //fp64_t nflops = m * n * k;
    // Submit task
    int ret;
    // dst is initialized by the codelet if beta is zero
    if(beta == 0.0)
    {
        ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(src1),
            STARPU_R, static_cast<starpu_data_handle_t>(src2),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_W, static_cast<starpu_data_handle_t>(dst),
            //STARPU_FLOPS, nflops,
            0);
    }
    // dst must be initialized before the codelet if beta is non-zero
    else
    {
        ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(src1),
            STARPU_R, static_cast<starpu_data_handle_t>(src2),
            STARPU_CL_ARGS, args, sizeof(*args),
            Config::STARPU_RW_COMMUTE, static_cast<starpu_data_handle_t>(dst),
            //STARPU_FLOPS, nflops,
            0);
    }
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in scalprod task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index m, Index n, Index k, fp32_t alpha, Handle src1,
        Handle src2, fp32_t beta, Handle dst);

template
void submit<fp64_t>(Index m, Index n, Index k, fp64_t alpha, Handle src1,
        Handle src2, fp64_t beta, Handle dst);

} // namespace scalprod
} // namespace starpu
} // namespace nntile

