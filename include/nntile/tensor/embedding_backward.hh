/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/embedding_backward.hh
 * Backward embeddings from vocabulary within Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Tensor-wise embedding_backward operation
template<typename T>
void embedding_backward_async(const Tensor<int64_t> &index,
        const Tensor<T> &vocab, const Tensor<T> &embed, Index axis,
        int redux=0);

// Tensor-wise embedding_backward operation
template<typename T>
void embedding_backward(const Tensor<int64_t> &index, const Tensor<T> &vocab,
        const Tensor<T> &embed, Index axis, int redux=0);

} // namespace nntile::tensor
