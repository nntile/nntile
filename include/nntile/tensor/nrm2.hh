/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/nrm2.hh
 * Euclidean norm of Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-05
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

//! Compute Euclidean norm
template<typename T>
void nrm2_async(T alpha, const Tensor<T> &src, T beta, const Tensor<T> &dst,
        const Tensor<T> &tmp);

template<typename T>
void nrm2(T alpha, const Tensor<T> &src, T beta, const Tensor<T> &dst,
        const Tensor<T> &tmp);

} // namespace tensor
} // namespace nntile

