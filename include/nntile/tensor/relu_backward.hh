/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/relu_backward.hh
 * Backward ReLU operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-04
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void relu_backward_async(const Tensor<T> &x, const Tensor<T> &dy,
        const Tensor<T> &dx);

template<typename T>
void relu_backward(const Tensor<T> &x, const Tensor<T> &dy,
        const Tensor<T> &dx);

} // namespace tensor
} // namespace nntile

