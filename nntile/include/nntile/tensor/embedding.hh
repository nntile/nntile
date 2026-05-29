/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/embedding.hh
 * Embeddings from vocabulary within Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Tensor-wise embedding operation
template<typename T>
void embedding_async(const Tensor<int64_t> &index, const Tensor<T> &vocab,
        const Tensor<T> &embed, Index axis);

// Tensor-wise embedding operation
template<typename T>
void embedding(const Tensor<int64_t> &index, const Tensor<T> &vocab,
        const Tensor<T> &embed, Index axis);

} // namespace nntile::tensor
