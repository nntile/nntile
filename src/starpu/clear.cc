/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/clear.cc
 * Clear a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/clear.hh"

// Standard libraries
#include <cstring>
#include <stdexcept>

namespace nntile::starpu
{

//! Constructor
Clear::Clear():
    codelet("nntile_clear", footprint, cpu_funcs, cuda_funcs)
{
    codelet.set_modes_fixed({STARPU_W});
}

//! Clear a StarPU buffer on CPU
void Clear::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    void *data = interfaces[0]->get_ptr<void>();
    // Clear buffer
    std::memset(data, 0, args->nbytes);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Clear a StarPU buffer on CUDA
void Clear::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    void *data = interfaces[0]->get_ptr<void>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Clear buffer
    cudaMemsetAsync(data, 0, args->nbytes, stream);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for clear tasks that depends only on cl_arg
uint32_t Clear::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nbytes, sizeof(args->nbytes), hash);
    return hash;
}

//! Submit clear task
void Clear::submit(Handle data)
{
    // Get number of bytes
    std::size_t nbytes = starpu_variable_get_elemsize(data.get());
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nbytes = nbytes;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_W, data.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in clear task submission");
    }
}

//! Clear operation object
Clear clear;

} // namespace nntile::starpu
