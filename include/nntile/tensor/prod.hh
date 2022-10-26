/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/prod.hh
 * Per-element product of two Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-10-26
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

// Asynchronous tensor-wise prod operation
template<typename T>
void prod_async(const Tensor<T> &src, const Tensor<T> &dst);

// Blocking version of tensor-wise prod operation
template<typename T>
void prod(const Tensor<T> &src, const Tensor<T> &dst);

} // namespace tensor
} // namespace nntile

