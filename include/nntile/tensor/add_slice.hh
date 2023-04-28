/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/add_slice.hh
 * Tensor wrappers for addition of a tensor and a broadcasted slice
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-28
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

// Tensor<T> addition of a tensor and a broadcasted slice
template<typename T>
void add_slice_async(T alpha, const Tensor<T> &src, T beta,
        const Tensor<T> &dst, Index axis);

// Tensor<T> addition of a tensor and a broadcasted slice
template<typename T>
void add_slice(T alpha, const Tensor<T> &src, T beta, const Tensor<T> &dst,
        Index axis);

} // namespace tensor
} // namespace nntile

