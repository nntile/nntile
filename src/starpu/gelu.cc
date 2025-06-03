/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/gelu.cc
 * GeLU operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/gelu.hh"

// Standard libraries
#include <cstdlib>

// Other NNTile headers
#include "nntile/kernel/gelu.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
Gelu<std::tuple<T>>::Gelu():
    codelet("nntile_gelu", footprint, cpu_funcs, cuda_funcs)
{
    codelet.set_modes_fixed({STARPU_RW});
}

//! Apply gelu on StarPU buffer on CPU
template<typename T>
void Gelu<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    // Launch kernel
    kernel::gelu::cpu<T>(args->nelems, data);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply gelu on StarPU buffer on CUDA
template<typename T>
void Gelu<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::gelu::cuda<T>(stream, args->nelems, data);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Submit gelu task
template<typename T>
void Gelu<std::tuple<T>>::submit(
    Index nelems,
    Handle data
)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_RW, data.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in gelu task submission");
    }
}

//! Pack of gelu operations for different types
gelu_pack_t gelu;

} // namespace nntile::starpu
