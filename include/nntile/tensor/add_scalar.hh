/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/add_scalar.hh
 * Add_scalar operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-05-09
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

// Tensor-wise add_scalar operation
template<typename T>
void add_scalar_async(T alpha, T beta, const Tensor<T> &dst);

// Tensor-wise add_scalar operation
template<typename T>
void add_scalar(T alpha, T beta, const Tensor<T> &dst);

} // namespace tensor
} // namespace nntile

