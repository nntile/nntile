/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
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
 * @author Aleksandr Katrutsa
 * @date 2023-02-11
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
template<typename T>
void bias_async(T alpha, const Tensor<T> &src);

// Tensor-wise bias operation
template<typename T>
void bias(const Tensor<T> &src, const Tensor<T> &dst, Index axis);
template<typename T>
void bias(T alpha, const Tensor<T> &src);

} // namespace tensor
} // namespace nntile

