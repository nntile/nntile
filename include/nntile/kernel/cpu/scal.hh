/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/cpu/scal.hh
 * Scaling operation for buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{

template<typename T>
void scal_kernel_cpu(Index nelems, T alpha, T *src)
    noexcept;

template<typename T>
void scal_starpu_cpu(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

