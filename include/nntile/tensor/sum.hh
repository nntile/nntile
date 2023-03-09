/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/sum.hh
 * Sum of Tensor<T> along axis
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author K. Sozykin
 * @date 2023-03-09
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void sum_async(const Tensor<T> &src, const Tensor<T> &dst, Index axis);

template<typename T>
void sum(const Tensor<T> &src, const Tensor<T> &dst, Index axis);

} // namespace tensor
} // namespace nntile

