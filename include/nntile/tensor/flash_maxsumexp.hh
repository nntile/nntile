/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/flash_maxsumexp.hh
 * Fast max and sum of exponents of Tensor<T> along axis
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-23
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void flash_maxsumexp_async(const Tensor<T> &Q, const Tensor<T> &K,
        const Tensor<bool_t> &mask, const Tensor<T> &maxsumexp,
        const Tensor<T> &tmp, int redux);

template<typename T>
void flash_maxsumexp(const Tensor<T> &Q, const Tensor<T> &K,
        const Tensor<bool_t> &mask, const Tensor<T> &maxsumexp,
        const Tensor<T> &tmp, int redux);

} // namespace tensor
} // namespace nntile

