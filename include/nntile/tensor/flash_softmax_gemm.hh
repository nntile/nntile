/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/flash_softmax_gemm.hh
 * Fast fused softmax and gemm operations
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-24
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void flash_softmax_gemm_async(const Tensor<T> &Q, const Tensor<T> &K,
        const Tensor<T> &V, const Tensor<bool_t> &mask,
        const Tensor<T> &maxsumexp, const Tensor<T> &dst,
        const Tensor<T> &tmp, int redux);

template<typename T>
void flash_softmax_gemm(const Tensor<T> &Q, const Tensor<T> &K,
        const Tensor<T> &V, const Tensor<bool_t> &mask,
        const Tensor<T> &maxsumexp, const Tensor<T> &dst,
        const Tensor<T> &tmp, int redux);


} // namespace tensor
} // namespace nntile

