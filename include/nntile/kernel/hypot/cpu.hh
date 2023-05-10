/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/hypot/cpu.hh
 * Hypot of 2 inputs
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-18
 * */

#pragma once

namespace nntile
{
namespace kernel
{
namespace hypot
{

template<typename T>
void cpu(T alpha, const T *x, T beta, T *y)
    noexcept;

} // namespace hypot
} // namespace kernel
} // namespace nntile

