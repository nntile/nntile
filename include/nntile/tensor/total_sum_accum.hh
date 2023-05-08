/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/total_sum_accum.hh
 * Total sum accumulating of Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-16
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void total_sum_accum_async(const Tensor<T> &logsumexp,
                           const Tensor<T> &src, const Tensor<Index> &class_labels,
                           const Tensor<T> &val);

template<typename T>
void total_sum_accum(const Tensor<T> &logsumexp,
                           const Tensor<T> &src, const Tensor<Index> &class_labels,
                           const Tensor<T> &val);

} // namespace tensor
} // namespace nntile

