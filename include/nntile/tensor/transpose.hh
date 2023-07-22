/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/transpose.hh
 * Transpose operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-20
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

// Tensor-wise transpose operation
template<typename T>
void transpose_async(T alpha, const Tensor<T> &src, const Tensor<T> &dst,
        Index ndim);

// Tensor-wise transpose operation
template<typename T>
void transpose(T alpha, const Tensor<T> &src, const Tensor<T> &dst,
        Index ndim);

} // namespace tensor
} // namespace nntile

