/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/rope_backward.cc
 * Backward of rotary positional embedding
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/rope_backward.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/rope_backward.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
RopeBackward<std::tuple<T>>::RopeBackward():
    codelet("nntile_rope_backward", footprint, cpu_funcs, cuda_funcs)
{
}

//! StarPU wrapper for kernel::rope_backward::cpu<T>
template<typename T>
void RopeBackward<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *sin = interfaces[0]->get_ptr<T>();
    const T *cos = interfaces[1]->get_ptr<T>();
    const T *dy = interfaces[2]->get_ptr<T>();
    T *dx = interfaces[3]->get_ptr<T>();
    // Launch kernel
    kernel::rope_backward::cpu<T>(args->m, args->n, sin, cos, dy, dx);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::rope_backward::cuda<T>
template<typename T>
void RopeBackward<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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
    kernel::rope_backward::cuda<T>(stream, args->m, args->n, sin, cos, src,
        dst);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for rope_backward tasks
template<typename T>
uint32_t RopeBackward<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // Apply hash over parameters m, and k
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    return hash;
}

template<typename T>
void RopeBackward<std::tuple<T>>::submit(Index m, Index n, Handle sin, Handle cos, Handle dy, Handle dx)
//! Insert rope_backward task into StarPU pool of tasks
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
    int ret = starpu_task_insert(&codelet,
            STARPU_R, sin.get(),
            STARPU_R, cos.get(),
            STARPU_R, dy.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_W, dx.get(),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in rope_backward task submission");
    }
}

//! Pack of rope_backward operations for different types
rope_backward_pack_t rope_backward;

} // namespace nntile::starpu
