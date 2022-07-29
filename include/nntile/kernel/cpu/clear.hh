/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/cpu/clear.hh
 * Clear a buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <cstddef>

namespace nntile
{

void clear_kernel_cpu(std::size_t size, void *buffer)
    noexcept;

void clear_starpu_cpu(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

