/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/gelutanh_backward.hh
 * Backward approximate GeLU operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-04-05
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void gelutanh_backward_async(const Tensor<T> &x, const Tensor<T> &dy,
        const Tensor<T> &dx);

template<typename T>
void gelutanh_backward(const Tensor<T> &x, const Tensor<T> &dy,
        const Tensor<T> &dx);

} // namespace tensor
} // namespace nntile

