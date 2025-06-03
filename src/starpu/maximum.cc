/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/maximum.cc
 * StarPU wrappers for maximum operation
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/maximum.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/maximum.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
Maximum<std::tuple<T>>::Maximum():
    codelet("nntile_maximum", footprint, cpu_funcs, cuda_funcs)
{
}

//! Apply maximum operation on StarPU buffers on CPU
template<typename T>
void Maximum<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    args_t *args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::maximum::cpu<T>(args->nelems, src, dst);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply per element maximum on StarPU buffer on CUDA
template<typename T>
void Maximum<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    args_t *args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::maximum::cuda<T>(stream, args->nelems, src, dst);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

template<typename T>
void Maximum<std::tuple<T>>::submit(Index nelems, Handle src, Handle dst)
{
    args_t *args = (args_t *)malloc(sizeof(*args));
    args->nelems = nelems;
    int ret = starpu_task_insert(&codelet,
            STARPU_R, src.get(),
            STARPU_RW, dst.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in maximum task submission");
    }
}

//! Pack of maximum operations for different types
maximum_pack_t maximum;

} // namespace nntile::starpu
