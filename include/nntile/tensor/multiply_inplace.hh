/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/multiply_inplace.hh
 * Per-element multiplication of two Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Asynchronous tensor-wise multiply operation
template<typename T>
void multiply_inplace_async(const Tensor<T> &src, const Tensor<T> &dst);

// Blocking version of tensor-wise multiply operation
template<typename T>
void multiply_inplace(const Tensor<T> &src, const Tensor<T> &dst);

} // namespace nntile::tensor
