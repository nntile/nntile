/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/gelutanh_backward/cpu.hh
 * Backward approximate GeLU operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-04-05
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace gelutanh_backward
{

// Approximate GeLU backward operation on a buffer
template<typename T>
void cpu(Index nelems, const T *x, const T *dy, T *dx)
    noexcept;

} // namespace gelutanh_backward
} // namespace kernel
} // namespace nntile

