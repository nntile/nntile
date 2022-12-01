/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
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
 * @date 2022-12-01
 * */

#pragma once

namespace nntile
{
namespace kernel
{
namespace hypot
{

template<typename T>
void cpu(const T *x, T *y)
    noexcept;

} // namespace hypot
} // namespace kernel
} // namespace nntile

