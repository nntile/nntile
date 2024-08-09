/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/subtract_indexed_outputs.hh
 * Subtraction of value from certain elements in Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

template<typename T>
void subtract_indexed_outputs_async(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<T> &dst);

template<typename T>
void subtract_indexed_outputs(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<T> &dst);

} // namespace nntile::tensor
