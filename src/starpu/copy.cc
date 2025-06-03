/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/copy.cc
 * Copy StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/copy.hh"

// Standard libraries
#include <cstring>

namespace nntile::starpu
{

//! Constructor
Copy::Copy():
    codelet("nntile_copy", nullptr, cpu_funcs, cuda_funcs)
{
    codelet.set_modes_fixed({STARPU_R, STARPU_W});
}

//! Copy StarPU buffers on CPU
void Copy::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const void *src = interfaces[0]->get_ptr<void>();
    void *dst = interfaces[1]->get_ptr<void>();
    // Launch kernel
    std::memcpy(dst, src, args->nbytes);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Copy StarPU buffers on CUDA
void Copy::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const void *src = interfaces[0]->get_ptr<void>();
    void *dst = interfaces[1]->get_ptr<void>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    cudaMemcpyAsync(dst, src, args->nbytes, cudaMemcpyDeviceToDevice, stream);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

void Copy::submit(Handle src, Handle dst)
//! Insert copy task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, src.get(),
            STARPU_W, dst.get(),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in copy task submission");
    }
}

} // namespace nntile::starpu
