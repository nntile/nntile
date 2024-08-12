/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/prod_fiber3.hh
 * Tensor wrappers for per-element product of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Tensor<T> per-element multiplication of a tensor and a broadcasted fiber
template<typename T>
void prod_fiber3_async(const Tensor<T> &src1, Scalar alpha, const Tensor<T> &src2,
        const Tensor<T> &dst, Index axis);

// Tensor<T> per-element multiplication of a tensor and a broadcasted fiber
template<typename T>
void prod_fiber3(const Tensor<T> &src1, Scalar alpha, const Tensor<T> &src2,
        const Tensor<T> &dst, Index axis);

} // namespace nntile::tensor
