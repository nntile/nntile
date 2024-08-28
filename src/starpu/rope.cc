/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/rope.cc
 * Rotary positional embedding
 *
 * @version 1.1.0
 * */

#ifndef STARPU_SIMGRID
#include "nntile/kernel/rope.hh"
#endif // STARPU_SIMGRID
#include "nntile/starpu/rope.hh"
#include <cstdlib>

//! StarPU wrappers for rope operation
namespace nntile::starpu::rope
{

//! StarPU wrapper for kernel::rope::cpu<T>
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *sin = interfaces[0]->get_ptr<T>();
    const T *cos = interfaces[1]->get_ptr<T>();
    const T *src = interfaces[2]->get_ptr<T>();
    T *dst = interfaces[3]->get_ptr<T>();
    // Launch kernel
    kernel::rope::cpu<T>(args->m, args->n, sin, cos, src, dst);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::rope::cuda<T>
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t*>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *sin = interfaces[0]->get_ptr<T>();
    const T *cos = interfaces[1]->get_ptr<T>();
    const T *src = interfaces[2]->get_ptr<T>();
    T *dst = interfaces[3]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::rope::cuda<T>(stream, args->m, args->n, sin, cos, src, dst);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for rope tasks
static
uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // Apply hash over parameters m, and k
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    return hash;
}

Codelet codelet_fp32, codelet_fp64, codelet_fp32_fast_tf32, codelet_bf16;

void init()
{
    codelet_fp32.init("nntile_rope_fp32",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp64.init("nntile_rope_fp64",
            footprint,
            {cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp64_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_tf32.init("nntile_rope_fp32_fast_tf32",
            footprint,
            {cpu<fp32_fast_tf32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_fast_tf32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_bf16.init("nntile_rope_bf16",
            footprint,
            {cpu<bf16_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<bf16_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
}

void restrict_where(uint32_t where)
{
    codelet_fp32.restrict_where(where);
    codelet_fp64.restrict_where(where);
    codelet_fp32_fast_tf32.restrict_where(where);
    codelet_bf16.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_fp64.restore_where();
    codelet_fp32_fast_tf32.restore_where();
    codelet_bf16.restore_where();
}

template<typename T>
void submit(Index m, Index n, Handle sin, Handle cos, Handle src, Handle dst)
//! Insert rope task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(sin),
            STARPU_R, static_cast<starpu_data_handle_t>(cos),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_W, static_cast<starpu_data_handle_t>(dst),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in rope task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index m, Index n, Handle sin, Handle cos, Handle src,
        Handle dst);

template
void submit<fp64_t>(Index m, Index n, Handle sin, Handle cos, Handle src,
        Handle dst);

template
void submit<fp32_fast_tf32_t>(Index m, Index n, Handle sin, Handle cos, Handle src,
        Handle dst);

template
void submit<bf16_t>(Index m, Index n, Handle sin, Handle cos, Handle src,
        Handle dst);

} // namespace rope
