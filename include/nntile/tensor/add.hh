/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/add.hh
 * Add operation for Tensor<T>'s
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-05-08
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

// Tensor-wise add operation
template<typename T>
void add_async(T alpha, const Tensor<T> &src, T beta, const Tensor<T> &dst);

// Tensor-wise add operation
template<typename T>
void add(T alpha, const Tensor<T> &src, T beta, const Tensor<T> &dst);

} // namespace tensor
} // namespace nntile

