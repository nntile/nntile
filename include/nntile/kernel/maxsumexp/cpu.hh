/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/maxsumexp/cpu.hh
 * Max and sum of exponents of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-12-07
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace maxsumexp
{

// Compute max and sums of exponents along middle axis
template<typename T>
void cpu(Index m, Index n, Index k, const T *src, T *maxsumexp)
    noexcept;

} // namespace maxsumexp
} // namespace kernel
} // namespace nntile

