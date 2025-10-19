/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/prod_slice.hh
 * Bias-like product operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Tensor-wise prod_slice operation
template<typename T>
void prod_slice_async(const Tensor<T> &src, Scalar alpha, const Tensor<T> &dst,
        Index axis);

// Tensor-wise prod_slice operation
template<typename T>
void prod_slice(const Tensor<T> &src, Scalar alpha, const Tensor<T> &dst,
        Index axis);

} // namespace nntile::tensor
