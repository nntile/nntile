/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/cpu/normalize.hh
 * Normalize operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-08
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace cpu
{

template<typename T>
void normalize(Index m, Index n, Index k, Index l, T eps, T gamma, T beta,
        const T *sumnorm, T *dst)
    noexcept;

} // namespace cpu
} // namespace kernel
} // namespace nntile

