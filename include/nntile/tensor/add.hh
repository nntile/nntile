/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/add.hh
 * Add operation for Tensor<T>'s
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Tensor-wise add operation
template<typename T>
void add_async(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst);

// Tensor-wise add operation
template<typename T>
void add(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst);

} // namespace nntile::tensor
