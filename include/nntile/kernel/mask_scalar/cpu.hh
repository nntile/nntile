/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/mask_scalar/cpu.hh
 * Mask operation with scalar on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-22
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace mask_scalar
{

// Mask scalar operation on a CPU buffer
template<typename T>
void cpu(Index nelems, bool_t* mask, T val, T *data)
    noexcept;

} // namespace mask_scalar
} // namespace kernel
} // namespace nntile

