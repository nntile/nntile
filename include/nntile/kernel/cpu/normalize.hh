/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/normalize.cc
 * Normalize operation for Tile<T>
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
void normalize_kernel_cpu(Index m, Index n, Index k, Index l, T eps, T gamma,
        T beta, const T *sumnorm, T *dst)
    noexcept;

template<typename T>
void normalize_starpu_cpu(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

