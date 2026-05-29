/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/embedding_backward.hh
 * Backward embeddings from vocabulary within Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

template<typename T>
void embedding_backward_async(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<T> &embed,
        const Tile<T> &vocab, int redux=0);

template<typename T>
void embedding_backward(Index m, Index n, Index k, Index k_start, Index k_size,
        const Tile<int64_t> &index, const Tile<T> &embed,
        const Tile<T> &vocab, int redux=0);

} // namespace nntile::tile
