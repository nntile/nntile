/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/norm.hh
 * Euclidean norm of all elements in a Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Tensor-wise norm
template<typename T>
void norm_async(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst);

// Tensor-wise norm
template<typename T>
void norm(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst);

// Mixed precision norm: compute norm of T tensor and store in U tensor
template<typename T, typename U>
void norm_async_mixed(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<U> &dst);

// Mixed precision norm: compute norm of T tensor and store in U tensor
template<typename T, typename U>
void norm_mixed(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<U> &dst);

} // namespace nntile::tensor
