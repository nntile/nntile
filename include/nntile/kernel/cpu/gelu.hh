/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/gelu.hh
 * GeLU operation
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{

// GeLU operation on a buffer
template<typename T>
void gelu_kernel_cpu(Index nelems, T *data)
    noexcept;

// GeLU operation on a StarPU buffer
template<typename T>
void gelu_starpu_cpu(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

