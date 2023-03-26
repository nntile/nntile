/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/scalprod.hh
 * Scalar products of two Tensor<T> along axis
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-03-26
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void scalprod_async(T alpha, const Tensor<T> &src1, const Tensor<T> &src2,
        T beta, const Tensor<T> &dst, Index axis);

template<typename T>
void scalprod(T alpha, const Tensor<T> &src1, const Tensor<T> &src2,
        T beta, const Tensor<T> &dst, Index axis);

} // namespace tensor
} // namespace nntile

