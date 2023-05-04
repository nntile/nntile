/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/copy.cc
 * Copy a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-04
 * */

#include "nntile/starpu/copy.hh"
#include <cstring>
#ifdef NNTILE_USE_CUDA
#   include <cuda_runtime.h>
#endif // NNTILE_USE_CUDA

#include <iostream>

namespace nntile
{
namespace starpu
{
namespace copy
{

//! Copy a StarPU buffer on CPU
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // No arguments
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    std::size_t size = interfaces[0]->elemsize;
    void *src = interfaces[0]->get_ptr<void>();
    void *dst = interfaces[1]->get_ptr<void>();
    // Copy buffer
    std::memcpy(dst, src, size);
}

#ifdef NNTILE_USE_CUDA
//! Copy a StarPU buffer on CUDA
void cuda(void *buffers[], void *cl_args)
    noexcept
{
    // No arguments
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    std::size_t size = interfaces[0]->elemsize;
    void *src = interfaces[0]->get_ptr<void>();
    void *dst = interfaces[1]->get_ptr<void>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Copy buffer
    cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToHost, stream);
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

//! Insert task to copy buffer
void submit(Handle src, Handle dst)
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

} // namespace copy
} // namespace starpu
} // namespace nntile

