/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/scale_fiber.hh
 * Tensor wrappers for scaling of a tensor with a broadcasted fiber
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Tensor<T> scaling of a tensor with a broadcasted fiber
template<typename T>
void scale_fiber_async(Scalar alpha, const Tensor<T> &src, const Tensor<T> &dst,
        Index axis, Index batch_ndim);

// Tensor<T> scaling of a tensor with a broadcasted fiber
template<typename T>
void scale_fiber(Scalar alpha, const Tensor<T> &src, const Tensor<T> &dst,
        Index axis, Index batch_ndim);

} // namespace nntile::tensor