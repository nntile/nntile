/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/subtract_indexed_column.hh
 * Subtraction of a given value from the indexed column in Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-18
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void subtract_indexed_column_async(T val,
                                   const Tensor<Index> &class_labels,
                                   const Tensor<T> &dst);

template<typename T>
void subtract_indexed_column(T val,
                             const Tensor<Index> &class_labels,
                             const Tensor<T> &dst);

} // namespace tensor
} // namespace nntile

