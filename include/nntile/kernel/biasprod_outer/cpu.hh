/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/biasprod_outer/cpu.hh
 * Bias-like product along outer axes operation on a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-20
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace biasprod_outer
{

// Apply biasprod_outer along outer axes on CPU
template<typename T>
void cpu(Index m, Index n, Index k, const T *src, T *dst)
    noexcept;

} // namespace biasprod_outer
} // namespace kernel
} // namespace nntile

