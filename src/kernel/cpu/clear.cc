/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/clear.cc
 * Clear a buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/kernel/cpu/clear.hh"
#include <cstring>
#include <starpu_data_interfaces.h>

namespace nntile
{

void clear_kernel_cpu(std::size_t size, void *buffer)
    noexcept
{
    memset(buffer, 0, size);
}

void clear_starpu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    auto ndim_buf = reinterpret_cast<starpu_ndim_interface *>(buffers[0]);
    std::size_t size = ndim_buf->allocsize;
    void *buffer = reinterpret_cast<void *>(ndim_buf->ptr);
    clear_kernel_cpu(size, buffer);
}

} // namespace nntile

