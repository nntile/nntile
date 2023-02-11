/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/bias/cpu.hh
 * Bias operation on a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-02-11
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace bias
{

// Apply bias along middle axis on CPU
template<typename T>
void cpu(Index m, Index n, Index k, const T *src, T *dst)
    noexcept;

template<typename T>
void cpu(T x, Index num_elements, T *y)
    noexcept;

} // namespace bias
} // namespace kernel
} // namespace nntile

