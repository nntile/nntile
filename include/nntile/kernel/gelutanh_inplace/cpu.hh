/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/gelutanh_inplace/cpu.hh
 * Approximate GeLU operation on CPU based on tanh function
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-01
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace gelutanh_inplace
{

// Approximate GeLU operation on a buffer on CPU
template<typename T>
void cpu(Index nelems, T *data)
    noexcept;

} // namespace gelutanh_inplace
} // namespace kernel
} // namespace nntile

