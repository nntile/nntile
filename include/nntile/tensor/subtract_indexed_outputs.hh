/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/subtract_indexed_outputs.hh
 * Subtraction of value from certain elements in Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-28
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void subtract_indexed_outputs_async(T val, const Tensor<Index> &labels,
        const Tensor<T> &dst);

template<typename T>
void subtract_indexed_outputs(T val, const Tensor<Index> &labels,
        const Tensor<T> &dst);

} // namespace tensor
} // namespace nntile

