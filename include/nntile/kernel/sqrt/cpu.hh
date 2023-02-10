/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sqrt/cpu.hh
 * sqrt operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-10
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace sqrt
{

// ReLU operation on a buffer
template<typename T>
void cpu(Index nelems, T *data)
    noexcept;

} // namespace sqrt
} // namespace kernel
} // namespace nntile

