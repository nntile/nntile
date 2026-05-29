/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/softmax.hh
 * Softmax operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

template<typename T>
void softmax_async(const Tensor<T> &maxsumexp, const Tensor<T> &src,
        Scalar alpha, const Tensor<T> &dst, Index axis);

template<typename T>
void softmax(const Tensor<T> &maxsumexp, const Tensor<T> &src,
        Scalar alpha, const Tensor<T> &dst, Index axis);

} // namespace nntile::tensor
