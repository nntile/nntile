/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/softmax.hh
 * Softmax operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-12-12
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void softmax_async(const Tensor<T> &maxsumexp, const Tensor<T> &dst,
        Index axis);

template<typename T>
void softmax(const Tensor<T> &maxsumexp, const Tensor<T> &dst, Index axis);

} // namespace tensor
} // namespace nntile

