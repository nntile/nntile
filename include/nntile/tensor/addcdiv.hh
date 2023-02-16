/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/addcdiv.hh
 * Addcdiv operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-14
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void addcdiv_async(T val, T eps, const Tensor<T> &nom, const Tensor<T> &denom,
                   const Tensor<T> &src);

template<typename T>
void addcdiv(T val, T eps, const Tensor<T> &nom, const Tensor<T> &denom,
             const Tensor<T> &src);

} // namespace tensor
} // namespace nntile

