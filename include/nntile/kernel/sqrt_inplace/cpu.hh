/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sqrt_inplace/cpu.hh
 * Inplace sqrt operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-07-01
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace sqrt_inplace
{

// Inplace sqrt operation on a buffer
template<typename T>
void cpu(Index nelems, T *data)
    noexcept;

} // namespace sqrt_inplace
} // namespace kernel
} // namespace nntile

