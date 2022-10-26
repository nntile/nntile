/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/prod/cpu.hh
 * Per-element product of two buffers on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-10-26
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace prod
{

// Per-element product of two buffers
template<typename T>
void cpu(Index nelems, const T *src, T *dst)
    noexcept;

} // namespace prod
} // namespace kernel
} // namespace nntile

