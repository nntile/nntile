/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/relu_backward/cpu.hh
 * Backward ReLU operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-04
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace relu_backward
{

// ReLU operation on a buffer
template<typename T>
void cpu(Index nelems, const T *x, const T *dy, T *dx)
    noexcept;

} // namespace relu_backward
} // namespace kernel
} // namespace nntile

