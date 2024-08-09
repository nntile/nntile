/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/add_scalar.hh
 * Add_scalar operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Tensor-wise add_scalar operation
template<typename T>
void add_scalar_async(Scalar alpha, Scalar beta, const Tensor<T> &dst);

// Tensor-wise add_scalar operation
template<typename T>
void add_scalar(Scalar alpha, Scalar beta, const Tensor<T> &dst);

} // namespace nntile::tensor
