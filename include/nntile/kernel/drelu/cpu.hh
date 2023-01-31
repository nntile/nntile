/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/drelu/cpu.hh
 * Derivative of ReLU operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-01-31
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace drelu
{

template<typename T>
void cpu(Index nelems, T *data)
    noexcept;

} // namespace drelu
} // namespace kernel
} // namespace nntile

