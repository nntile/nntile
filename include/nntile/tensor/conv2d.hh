/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/conv2d.hh
 * Tensor wrappers for 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-03
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

// Tensor<T> 2D-Convolution between 2 matrices
template <typename T>
void conv2d_async(const Tensor<T> &src, const Tensor<T> &kernel,
                  const Tensor<T> &dst);

// Tensor<T> 2D-Convolution between 2 matrices
template <typename T>
void conv2d(const Tensor<T> &src, const Tensor<T> &kernel,
            const Tensor<T> &dst);

} // namespace tensor
} // namespace nntile
