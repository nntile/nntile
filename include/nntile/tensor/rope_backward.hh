/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/rope_backward.hh
 * Tensor wrappers for the Rotary Positional Embedding
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Tensor<T> RoPE
template<typename T>
void rope_backward_async(const Tensor<T> &sin, const Tensor<T> &cos,
        const Tensor<T> &dy, const Tensor<T> &dx);

// Tensor<T> RoPE
template<typename T>
void rope_backward(const Tensor<T> &sin, const Tensor<T> &cos, const Tensor<T> &dy,
        const Tensor<T> &dx);

} // namespace nntile::tensor
