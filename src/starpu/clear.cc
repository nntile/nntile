/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/clear.cc
 * Clear a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-31
 * */

#include "nntile/starpu/clear.hh"
#include <cstring>
#ifdef NNTILE_USE_CUDA
#   include <cuda_runtime.h>
#endif // NNTILE_USE_CUDA

#include <iostream>

namespace nntile
{
namespace starpu
{
namespace clear
{

//! Clear a StarPU buffer on CPU
void clear_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // No arguments
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    std::size_t size = interfaces[0]->elemsize;
    void *data = interfaces[0]->get_ptr<void>();
    // Clear buffer
    std::memset(data, 0, size);
}

#ifdef NNTILE_USE_CUDA
//! Clear a StarPU buffer on CUDA
void clear_cuda(void *buffers[], void *cl_args)
    noexcept
{
    // No arguments
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    std::size_t size = interfaces[0]->elemsize;
    void *data = interfaces[0]->get_ptr<void>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Clear buffer
    cudaMemsetAsync(data, 0, size, stream);
}
#endif // NNTILE_USE_CUDA

StarpuCodelet clear_codelet;

void clear_init()
{
    clear_codelet.init("nntile_clear",
            nullptr,
            {clear_cpu},
#ifdef NNTILE_USE_CUDA
            {clear_cuda}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
}

void clear_restrict_where(uint32_t where)
{
    clear_codelet.restrict_where(where);
}

void clear_restore_where()
{
    clear_codelet.restore_where();
}

//! Insert task to clear buffer
void clear(starpu_data_handle_t data)
{
    // Submit task
    int ret = starpu_task_insert(&clear_codelet,
            STARPU_W, data,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in clear task submission");
    }
}

} // namespace clear
} // namespace starpu
} // namespace nntile

