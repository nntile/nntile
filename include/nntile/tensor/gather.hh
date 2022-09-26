/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/gather.hh
 * Gather operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-26
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

// Asynchronous tensor-wise gather operation
template<typename T>
void gather_async(const Tensor<T> &src, const Tensor<T> &dst);

// Blocking version of tensor-wise gather operation
template<typename T>
void gather(const Tensor<T> &src, const Tensor<T> &dst);

} // namespace tensor
} // namespace nntile

