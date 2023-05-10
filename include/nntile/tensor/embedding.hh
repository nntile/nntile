/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/embedding.hh
 * Embeddings from vocabulary within Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-21
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

// Tensor-wise embedding operation
template<typename T>
void embedding_async(const Tensor<Index> &index, const Tensor<T> &vocab,
        const Tensor<T> &embed, Index axis);

// Tensor-wise embedding operation
template<typename T>
void embedding(const Tensor<Index> &index, const Tensor<T> &vocab,
        const Tensor<T> &embed, Index axis);

} // namespace tensor
} // namespace nntile

