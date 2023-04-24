/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/fill.hh
 * Fill operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-24
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void fill_async(T val, const Tensor<T> &A);

template<typename T>
void fill(T val, const Tensor<T> &A);

} // namespace tensor
} // namespace nntile

