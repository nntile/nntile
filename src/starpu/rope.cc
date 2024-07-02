/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/rope.cc
 * StarPU wrappers for addition of a tensor and a broadcasted fiber
 *
 * @version 1.0.0
 * @author Gleb Karpov
 * @date 2024-05-27
 * */

#include "nntile/starpu/rope.hh"
#include "nntile/kernel/rope.hh"
#include <cstdlib>

namespace nntile
{
namespace starpu
{
//! StarPU wrappers for add_fiber operation
namespace rope
{

//! StarPU wrapper for kernel::add_fiber::cpu<T>
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *sin = interfaces[0]->get_ptr<T>();
    const T *cos = interfaces[1]->get_ptr<T>();
    const T *src = interfaces[2]->get_ptr<T>();
    T *dst = interfaces[3]->get_ptr<T>();
    // Launch kernel
    kernel::rope::cpu<T>(args->m, args->k, args->l, sin, cos, src, dst);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::rope::cuda<T>
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
    // // Get arguments
    // auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // // Get interfaces
    // auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    // const T *src = interfaces[0]->get_ptr<T>();
    // T *dst = interfaces[1]->get_ptr<T>();
    // // Get CUDA stream
    // cudaStream_t stream = starpu_cuda_get_local_stream();
    // // Launch kernel
    // kernel::add_fiber::cuda<T>(stream, args->m, args->n, args->k, args->batch,
    //         args->alpha, src, args->beta, dst);
}
#endif // NNTILE_USE_CUDA

//! Footprint for rope tasks
template<typename T>
static
uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(task->cl_arg);
    // Apply hash over parameters m, and k
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    hash = starpu_hash_crc32c_be_n(&args->l, sizeof(args->l), hash);
    return hash;
}

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_rope_fp32",
            footprint<fp32_t>,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_rope_fp64",
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
void submit(Index m, Index k, Index l, Handle sin, Handle cos,
    Handle src, Handle dst)
//! Insert rope task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Access mode for the dst handle
    enum starpu_data_access_mode dst_mode;
    dst_mode = STARPU_RW;

    // Codelet arguments
    args_t<T> *args = (args_t<T> *)std::malloc(sizeof(*args));
    args->m = m;
    args->k = k;
    args->l = l;
    // fp64_t nflops = batch * k * (2*m*n+1);
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(sin),
            STARPU_R, static_cast<starpu_data_handle_t>(cos),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_W, static_cast<starpu_data_handle_t>(dst),
            // STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in rope task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index m, Index k, Index l, Handle sin,
        Handle cos, Handle src, Handle dst);

template
void submit<fp64_t>(Index m, Index k, Index l, Handle sin,
        Handle cos, Handle src, Handle dst);

} // namespace rope
} // namespace starpu
} // namespace nntile