/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/hypot_scalar_inverse/cpu.hh
 * Inverse of a hypot operation of a buffer and a scalar
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-28
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace hypot_scalar_inverse
{

template<typename T>
void cpu(Index nelems, T eps, T alpha, T* dst)
    noexcept;

} // namespace hypot_scalar_inverse
} // namespace kernel
} // namespace nntile

