/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/cpu/sumnorm.hh
 * Sum and Euclidian norm of a buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{

// Compute sum and Euclidian norm along middle axis
template<typename T>
void sumnorm_kernel_cpu(Index m, Index n, Index k, const T *src, bool init,
        T *sumnorm)
    noexcept;

// Compute sum and Euclidian norm along middle axis of StarPU buffer
template<typename T>
void sumnorm_starpu_cpu(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

