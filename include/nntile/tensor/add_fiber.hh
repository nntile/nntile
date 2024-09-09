/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/add_fiber.hh
 * Tensor wrappers for addition of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Tensor<T> addition of a tensor and a broadcasted fiber
template<typename T>
void add_fiber_async(Scalar alpha, const Tensor<T> &src1, Scalar beta,
        const Tensor<T> &src2, const Tensor<T> &dst, Index axis, Index batch_ndim);

// Tensor<T> addition of a tensor and a broadcasted fiber
template<typename T>
void add_fiber(Scalar alpha, const Tensor<T> &src1, Scalar beta,
        const Tensor<T> &src2, const Tensor<T> &dst, Index axis, Index batch_ndim);

} // namespace nntile::tensor
