/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add/cpu.hh
 * Add operation on buffers on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-05-08
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace add
{

// Apply add for buffers on CPU
template<typename T>
void cpu(Index num_elements, T alpha, const T* src, T beta, T* dst)
    noexcept;

} // namespace add
} // namespace kernel
} // namespace nntile

