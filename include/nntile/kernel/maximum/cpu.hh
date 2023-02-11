/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/maximum/cpu.hh
 * Per-element maximum of two buffers on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-10
 * */

#pragma once

#include <nntile/base_types.hh>
#include <algorithm>

namespace nntile
{
namespace kernel
{
namespace maximum
{

// Per-element maximum of two buffers
template<typename T>
void cpu(Index nelems, const T *src, T *dst)
    noexcept;

} // namespace maximum
} // namespace kernel
} // namespace nntile

