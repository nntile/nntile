/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/pow/cpu.hh
 * Power operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-14
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace pow
{

// Power operation on a buffer
template<typename T>
void cpu(Index nelems, T alpha, T exp, T *data)
    noexcept;

} // namespace pow
} // namespace kernel
} // namespace nntile

