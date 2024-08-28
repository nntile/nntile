/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/scal.hh
 * Scal operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Tensor-wise scal operation
template<typename T>
void scal_async(Scalar alpha, const Tensor<T> &src, const Tensor<T> &dst);

// Tensor-wise scal operation
template<typename T>
void scal(Scalar alpha, const Tensor<T> &src, const Tensor<T> &dst);

} // namespace nntile::tensor
