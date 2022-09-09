/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/bias.hh
 * Bias operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-07
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

// Tensor-wise bias operation
template<typename T>
void bias_async(const Tensor<T> &src, const Tensor<T> &dst, Index axis);

// Tensor-wise bias operation
template<typename T>
void bias(const Tensor<T> &src, const Tensor<T> &dst, Index axis);

} // namespace tensor
} // namespace nntile

