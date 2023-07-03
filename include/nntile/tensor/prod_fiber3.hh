/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/prod_fiber3.hh
 * Tensor wrappers for per-element product of a tensor and a broadcasted fiber
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-03
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

// Tensor<T> per-element multiplication of a tensor and a broadcasted fiber
template<typename T>
void prod_fiber3_async(const Tensor<T> &src1, T alpha, const Tensor<T> &src2,
        const Tensor<T> &dst, Index axis);

// Tensor<T> per-element multiplication of a tensor and a broadcasted fiber
template<typename T>
void prod_fiber3(const Tensor<T> &src1, T alpha, const Tensor<T> &src2,
        const Tensor<T> &dst, Index axis);

} // namespace tensor
} // namespace nntile

