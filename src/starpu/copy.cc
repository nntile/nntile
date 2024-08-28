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

#include "nntile/starpu/copy.hh"
#include <cstring>

//! StarPU wrappers for copy operation
namespace nntile::starpu::copy
{

//! Copy StarPU buffers on CPU
void cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // No arguments
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    std::size_t size = interfaces[0]->elemsize;
    const void *src = interfaces[0]->get_ptr<void>();
    void *dst = interfaces[1]->get_ptr<void>();
    // Launch kernel
    std::memcpy(dst, src, size);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Copy StarPU buffers on CUDA
void cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // No arguments
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    std::size_t size = interfaces[0]->elemsize;
    const void *src = interfaces[0]->get_ptr<void>();
    void *dst = interfaces[1]->get_ptr<void>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

Codelet codelet;

void init()
{
    codelet.init("nntile_copy",
            nullptr,
            {cpu},
#ifdef NNTILE_USE_CUDA
            {cuda}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
}

void restrict_where(uint32_t where)
{
    codelet.restrict_where(where);
}

void restore_where()
{
    codelet.restore_where();
}

void submit(Handle src, Handle dst)
//! Insert copy task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            STARPU_W, static_cast<starpu_data_handle_t>(dst),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in copy task submission");
    }
}

} // namespace nntile::starpu::copy
