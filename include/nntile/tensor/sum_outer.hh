/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/sum_outer.hh
 * Sum of Tensor<T> along outer axes
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-19
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void sum_outer_async(T alpha, const Tensor<T> &src, T beta,
        const Tensor<T> &sum_dst, Index axis);

template<typename T>
void sum_outer(T alpha, const Tensor<T> &src, T beta, const Tensor<T> &sum_dst,
        Index axis);

} // namespace tensor
} // namespace nntile

