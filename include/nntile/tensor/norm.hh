/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/norm.hh
 * Euclidian norm of slices of a Tensor<T>
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
void norm_async(T alpha, const Tensor<T> &src, T beta,
        const Tensor<T> &norm_dst, Index axis);

template<typename T>
void norm(T alpha, const Tensor<T> &src, T beta, const Tensor<T> &norm_dst,
        Index axis);

} // namespace tensor
} // namespace nntile

