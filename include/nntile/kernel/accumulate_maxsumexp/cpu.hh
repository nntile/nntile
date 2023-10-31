/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/accumulate_maxsumexp/cpu.hh
 * Accumulate maxsumexp buffers on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-20
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace accumulate_maxsumexp
{

// Accumulate maxsumexp buffers on CPU
template<typename T>
void cpu(Index nelems, const T* src, T* dst)
    noexcept;

} // namespace accumulate_maxsumexp
} // namespace kernel
} // namespace nntile

