/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/clear.cc
 * Clear a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-02
 * */

#include "nntile/kernel/cpu/clear.hh"
#include "nntile/starpu.hh"
#include <cstring>

namespace nntile
{

//! Clear buffer
//
// @param[in] size: size of buffer to clear in bytes
// @param[out] dst: buffer to fill with zeros
void clear_kernel_cpu(std::size_t size, void *dst)
    noexcept
{
    std::memset(dst, 0, size);
}

void clear_starpu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // No arguments
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    std::size_t size = interfaces[0]->elemsize;
    void *dst = interfaces[0]->get_ptr<void>();
    clear_kernel_cpu(size, dst);
}

} // namespace nntile

