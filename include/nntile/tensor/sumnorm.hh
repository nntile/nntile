/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/sumnorm.hh
 * Sum and Euclidean norm of Tensor<T> along axis
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-05
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void sumnorm_async(const Tensor<T> &src, const Tensor<T> &dst, Index axis);

template<typename T>
void sumnorm(const Tensor<T> &src, const Tensor<T> &dst, Index axis);

} // namespace tensor
} // namespace nntile

