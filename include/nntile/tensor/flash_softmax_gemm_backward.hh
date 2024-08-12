/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/flash_softmax_gemm_backward.hh
 * Fast backward of softmax and gemm operations
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

template<typename T>
void flash_softmax_gemm_backward_async(const Tensor<T> &Q, const Tensor<T> &dQ,
        const Tensor<T> &K, const Tensor<T> &dK, const Tensor<T> &V,
        const Tensor<T> &dV, const Tensor<bool_t> &mask,
        const Tensor<T> &maxsumexp, const Tensor<T> &dst_grad,
        const Tensor<T> &tmp, const Tensor<T> &tmp_grad,
        const Tensor<T> &tmp_sumprod_slice, int redux=0);

template<typename T>
void flash_softmax_gemm_backward(const Tensor<T> &Q, const Tensor<T> &dQ,
        const Tensor<T> &K, const Tensor<T> &dK, const Tensor<T> &V,
        const Tensor<T> &dV, const Tensor<bool_t> &mask,
        const Tensor<T> &maxsumexp, const Tensor<T> &dst_grad,
        const Tensor<T> &tmp, const Tensor<T> &tmp_grad,
        const Tensor<T> &tmp_sumprod_slice, int redux=0);

} // namespace nntile::tensor
