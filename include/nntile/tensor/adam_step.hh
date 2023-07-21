/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/adam_step.hh
 * Fused Adam step operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-07-21
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void adam_step_async(Index num_iter, T beta_1, T beta_2, T eps, T lr,
    const Tensor<T> &grad, const Tensor<T> &first_moment, const Tensor<T> &second_moment,
                   const Tensor<T> &p);

template<typename T>
void adam_step(Index num_iter, T beta_1, T beta_2, T eps, T lr,
    const Tensor<T> &grad, const Tensor<T> &first_moment, const Tensor<T> &second_moment,
                   const Tensor<T> &p);

} // namespace tensor
} // namespace nntile

