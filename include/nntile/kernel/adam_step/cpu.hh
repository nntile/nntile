/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/adam_step/cpu.hh
 * Fused Adam step on CPU buffers
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-07-21
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace adam_step
{


template<typename T>
void cpu(Index num_iter, Index num_elems, T beta_1, T beta_2, T eps, T lr, T weight_decay,
         T* grad, T* first_moment, T* second_moment, T* p)
    noexcept;

} // namespace adam_step
} // namespace kernel
} // namespace nntile