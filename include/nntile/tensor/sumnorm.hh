/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/sumnorm.hh
 * Sum and Euclidean norm of Tensor<T> along axis
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

template<typename T>
void sumnorm_async(const Tensor<T> &src, const Tensor<T> &dst, Index axis);

template<typename T>
void sumnorm(const Tensor<T> &src, const Tensor<T> &dst, Index axis);

} // namespace nntile::tensor
