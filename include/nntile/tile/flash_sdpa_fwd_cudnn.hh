/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/flash_sdpa_fwd_cudnn.hh
 * Flash attention scaled dot-product attention forward pass using cuDNN
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

// Asynchronous tile-wise flash_sdpa_fwd_cudnn operation
template<typename T>
void flash_sdpa_fwd_cudnn_async(const Tile<T> &K, const Tile<T> &Q,
        const Tile<T> &mask, const Tile<T> &logsumexp, const Tile<T> &V,
        const Tile<T> &A);

// Blocking version of tile-wise flash_sdpa_fwd_cudnn operation
template<typename T>
void flash_sdpa_fwd_cudnn(const Tile<T> &K, const Tile<T> &Q,
        const Tile<T> &mask, const Tile<T> &logsumexp, const Tile<T> &V,
        const Tile<T> &A);

} // namespace nntile::tile
