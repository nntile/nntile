/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/hypot.hh
 * hypot operation for Tensor<T>
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

// Tensor-wise hypot operation
template<typename T>
void hypot_async(T alpha, const Tensor<T> &src, T beta, const Tensor<T> &dst);

// Tensor-wise hypot operation
template<typename T>
void hypot(T alpha, const Tensor<T> &src, T beta, const Tensor<T> &dst);

} // namespace tensor
} // namespace nntile

