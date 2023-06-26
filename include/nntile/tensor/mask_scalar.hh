/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/mask_scalar.hh
 * Mask scalar operation on tensor 
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2022-06-22
 * */

#pragma once

#include "nntile/tensor/tensor.hh"

namespace nntile
{
namespace tensor
{

// Asynchronous tensor-wise mask_scalar operation
template<typename T>
void mask_scalar_async(const Tensor<bool_t> &mask, T val, const Tensor<T> &A);

// Blocking version of tensor-wise mask_scalar operation
template<typename T>
void mask_scalar(const Tensor<bool_t> &mask, T val, const Tensor<T> &A);

} // namespace tensor
} // namespace nntile

