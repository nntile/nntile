/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/hypot_scalar_inverse.hh
 * hypot_scalar_inverse operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

template<typename T>
void hypot_scalar_inverse_async(Scalar eps, Scalar alpha, const Tensor<T> &dst);

template<typename T>
void hypot_scalar_inverse(Scalar eps, Scalar alpha, const Tensor<T> &dst);

} // namespace nntile::tensor
