/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/embedding_backward/cpu.hh
 * Backward of embeddings from vocabulary within buffers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::embedding_backward
{

// Accumulate gradients of embeddings into vocabulary
template<typename T>
void cpu(Index m, Index n, Index k, Index k_start, Index k_size,
        const int64_t *index, const T *embed, T *vocab)
    noexcept;

} // namespace nntile::kernel::embedding_backward
