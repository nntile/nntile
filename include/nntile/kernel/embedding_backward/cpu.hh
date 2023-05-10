/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/embedding_backward/cpu.hh
 * Backward of embeddings from vocabulary within buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-21
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace embedding_backward
{

// Accumulate gradients of embeddings into vocabulary
template<typename T>
void cpu(Index m, Index n, Index k, Index k_start, Index k_size,
        const Index *index, const T *embed, T *vocab)
    noexcept;

} // namespace embedding_backward
} // namespace kernel
} // namespace nntile

