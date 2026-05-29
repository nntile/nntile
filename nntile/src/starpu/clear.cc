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
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace nntile::starpu
{

//! Constructor
Clear::Clear():
    codelet("nntile_clear", nullptr, cpu_funcs, cuda_funcs)
{
    // Modes cannot be variable for clear operation
    // Construct modes
    constexpr std::array<starpu_data_access_mode, 1> modes = {
        STARPU_W
    };
    // Set modes
    codelet.set_modes_fixed(modes);
}

//! Clear a StarPU buffer on CPU
void Clear::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    std::size_t nbytes = interfaces[0]->elemsize;
    void *data = interfaces[0]->get_ptr<void>();
    // Clear buffer
    std::memset(data, 0, nbytes);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Clear a StarPU buffer on CUDA
void Clear::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    std::size_t nbytes = interfaces[0]->elemsize;
    void *data = interfaces[0]->get_ptr<void>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Clear buffer
    cudaMemsetAsync(data, 0, nbytes, stream);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Submit clear task
void Clear::submit(Handle data)
{
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_W, data.get(),
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
