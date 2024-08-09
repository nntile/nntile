/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/mask_scalar.hh
 * Mask scalar operation on tensor
 *
 * @version 1.1.0
 * */

#pragma once

#include "nntile/tensor/tensor.hh"

namespace nntile::tensor
{

// Asynchronous tensor-wise mask_scalar operation
template<typename T>
void mask_scalar_async(const Tensor<bool_t> &mask, Scalar val, const Tensor<T> &A,
        Index batch_ndim);

// Blocking version of tensor-wise mask_scalar operation
template<typename T>
void mask_scalar(const Tensor<bool_t> &mask, Scalar val, const Tensor<T> &A,
        Index batch_ndim);

} // namespace nntile::tensor
