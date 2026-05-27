/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/clear.hh
 * Clear Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/tensor/tensor.hh>

namespace nntile::core::tensor
{

template<typename T>
void clear_async(const Tensor<T> &dst);

template<typename T>
void clear(const Tensor<T> &dst);

} // namespace nntile::core::tensor
