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

#include "nntile/starpu/clear.hh"
#include <cstring>
#ifndef STARPU_SIMGRID
#   ifdef NNTILE_USE_CUDA
#       include <cuda_runtime.h>
#   endif // NNTILE_USE_CUDA
#endif // STARPU_SIMGRID

#include <iostream>

namespace nntile::starpu::clear
{

//! Clear a StarPU buffer on CPU
void cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // No arguments
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    std::size_t size = interfaces[0]->elemsize;
    void *data = interfaces[0]->get_ptr<void>();
    // Clear buffer
    std::memset(data, 0, size);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Clear a StarPU buffer on CUDA
void cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // No arguments
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    std::size_t size = interfaces[0]->elemsize;
    void *data = interfaces[0]->get_ptr<void>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Clear buffer
    cudaMemsetAsync(data, 0, size, stream);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

Codelet codelet;

void init()
{
    codelet.init("nntile_clear",
            nullptr,
            {cpu},
#ifdef NNTILE_USE_CUDA
            {cuda}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet.nbuffers = 1;
    codelet.modes[0] = STARPU_W;
}

void restrict_where(uint32_t where)
{
    codelet.restrict_where(where);
}

void restore_where()
{
    codelet.restore_where();
}

//! Insert task to clear buffer
void submit(Handle data)
{
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_W, static_cast<starpu_data_handle_t>(data),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in clear task submission");
    }
}

} // namespace nntile::starpu::clear
