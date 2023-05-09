/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add_scalar/cpu.hh
 * Add_scalar operation on buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-05-09
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace add_scalar
{

// Apply add_scalar for buffer on CPU
template<typename T>
void cpu(Index num_elements, T alpha, T beta, T* dst)
    noexcept;

} // namespace add_scalar
} // namespace kernel
} // namespace nntile

