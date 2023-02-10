/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/maximum.hh
 * Per-element maximum of two Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-10
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

// Asynchronous tensor-wise maximum operation
template<typename T>
void maximum_async(const Tensor<T> &src, const Tensor<T> &dst);

// Blocking version of tensor-wise maximum operation
template<typename T>
void maximum(const Tensor<T> &src, const Tensor<T> &dst);

} // namespace tensor
} // namespace nntile
