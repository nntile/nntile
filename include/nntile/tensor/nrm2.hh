/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/nrm2.hh
 * Euclidian norm of Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-12-02
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

//! Compute Euclidian norm
template<typename T>
void nrm2_async(const Tensor<T> &src, const Tensor<T> &dst,
        const Tensor<T> &tmp);

template<typename T>
void nrm2(const Tensor<T> &src, const Tensor<T> &dst, const Tensor<T> &tmp);

} // namespace tensor
} // namespace nntile

