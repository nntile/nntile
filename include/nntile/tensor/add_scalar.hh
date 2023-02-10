/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/add_scalar.hh
 * Add scalar to elements from Tensor<T>
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

// Asynchronous tensor-wise add scalar operation
template<typename T>
void add_scalar_async(T alpha, const Tensor<T> &src);

// Blocking version of tensor-wise add_scalar operation
template<typename T>
void add_scalar(T alpha, const Tensor<T> &src);

} // namespace tensor
} // namespace nntile

