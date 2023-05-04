/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/relu_forward.hh
 * Forward ReLU operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-04
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void relu_forward_async(const Tensor<T> &src, const Tensor<T> &dst);

template<typename T>
void relu_forward(const Tensor<T> &src, const Tensor<T> &dst);

} // namespace tensor
} // namespace nntile

