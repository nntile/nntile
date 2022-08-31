/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/normalize/cpu.hh
 * Normalize operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-31
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace normalize
{

template<typename T>
void cpu(Index m, Index n, Index k, Index l, T eps, const T *gamma,
        const T *beta, const T *sumnorm, T *dst)
    noexcept;

} // namespace normalize
} // namespace kernel
} // namespace nntile

