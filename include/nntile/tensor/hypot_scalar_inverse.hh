/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/hypot_scalar_inverse.hh
 * hypot_scalar_inverse operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-29
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void hypot_scalar_inverse_async(T eps, T alpha, const Tensor<T> &dst);

template<typename T>
void hypot_scalar_inverse(T eps, T alpha, const Tensor<T> &dst);

} // namespace tensor
} // namespace nntile

