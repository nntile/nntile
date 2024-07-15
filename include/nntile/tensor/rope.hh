/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/rope.hh
 * Tensor wrappers for the Rotary Positional Embedding
 *
 * @version 1.0.0
 * @author Gleb Karpov
 * @date 2024-06-29
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Tensor<T> RoPE
template<typename T>
void rope_async(const Tensor<T> &sin, const Tensor<T> &cos, 
        const Tensor<T> &src, const Tensor<T> &dst, Index axis);

// Tensor<T> RoPE
template<typename T>
void rope(const Tensor<T> &sin, const Tensor<T> &cos, 
        const Tensor<T> &src, const Tensor<T> &dst, Index axis);

} // namespace tensor