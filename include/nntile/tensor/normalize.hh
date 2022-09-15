/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/normalize.hh
 * Normalize operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-15
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void normalize_async(const StarpuVariableHandle &gamma_beta,
        const Tensor<T> &src, const Tensor<T> &dst, Index l, T eps,
        Index axis);

template<typename T>
void normalize(const StarpuVariableHandle &gamma_beta,
        const Tensor<T> &src, const Tensor<T> &dst, Index l, T eps,
        Index axis);

} // namespace tensor
} // namespace nntile

