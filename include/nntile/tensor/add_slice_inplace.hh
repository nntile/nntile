/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/add_slice_inplace.hh
 * Tensor wrappers for addition of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Tensor<T> addition of a tensor and a broadcasted slice
template<typename T>
void add_slice_inplace_async(Scalar alpha, const Tensor<T> &src, Scalar beta,
        const Tensor<T> &dst, Index axis);

// Tensor<T> addition of a tensor and a broadcasted slice
template<typename T>
void add_slice_inplace(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst,
        Index axis);

} // namespace nntile::tensor
