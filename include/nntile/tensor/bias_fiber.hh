/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/bias_fiber.hh
 * Bias operation over slices from a fiber of a Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-26
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

// Tensor-wise bias_fiber operation
template<typename T>
void bias_fiber_async(T alpha, const Tensor<T> &src, T beta,
        const Tensor<T> &dst, Index axis);

// Tensor-wise bias_fiber operation
template<typename T>
void bias_fiber(T alpha, const Tensor<T> &src, T beta, const Tensor<T> &dst,
        Index axis);

} // namespace tensor
} // namespace nntile

