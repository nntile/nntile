/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/cuda/bias.cuh
 * Bias operation for Tile<T>
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
__global__
void bias_kernel_cuda(Index m, Index n, Index k, Index mk, const T *src,
        T *dst)
    noexcept;

template<typename T>
void bias_starpu_cuda(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

