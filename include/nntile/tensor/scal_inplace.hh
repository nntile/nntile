/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/scal_inplace.hh
 * Inplace scal of Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-02
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

//! scal_inplacee tensor
template<typename T>
void scal_inplace_async(T alpha, const Tensor<T> &data);

template<typename T>
void scal_inplace(T alpha, const Tensor<T> &data);

} // namespace tensor
} // namespace nntile

