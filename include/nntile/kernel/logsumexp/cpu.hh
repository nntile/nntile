/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/logsumexp/cpu.hh
 * Logsumexp of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-15
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace logsumexp
{

// Compute logsumexp based on the resut of maxsumexp operation 
template<typename T>
void cpu(Index m, const T *src, T *logsumexp)
    noexcept;

} // namespace logsumexp
} // namespace kernel
} // namespace nntile

