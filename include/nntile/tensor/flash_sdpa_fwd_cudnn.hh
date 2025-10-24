/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/flash_sdpa_fwd_cudnn.hh
 * Flash attention scaled dot-product attention forward pass using cuDNN
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Asynchronous tensor-wise flash_sdpa_fwd_cudnn operation
template<typename T>
void flash_sdpa_fwd_cudnn_async(const Tensor<T> &K, const Tensor<T> &Q,
        const Tensor<T> &mask, const Tensor<T> &logsumexp, const Tensor<T> &V,
        const Tensor<T> &A);

// Blocking version of tensor-wise flash_sdpa_fwd_cudnn operation
template<typename T>
void flash_sdpa_fwd_cudnn(const Tensor<T> &K, const Tensor<T> &Q,
        const Tensor<T> &mask, const Tensor<T> &logsumexp, const Tensor<T> &V,
        const Tensor<T> &A);

} // namespace nntile::tensor
