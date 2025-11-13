/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/flash_sdpa_bwd_cudnn.hh
 * Flash attention scaled dot-product attention backward pass using cuDNN
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

template<typename T>
void flash_sdpa_bwd_cudnn_async(
    const Tile<T> &K,
    const Tile<T> &Q,
    const Tile<T> &V,
    const Tile<T> &O,
    const Tile<T> &dO,
    const Tile<T> &mask,
    const Tile<fp32_t> &logsumexp,
    const Tile<T> &dK,
    const Tile<T> &dQ,
    const Tile<T> &dV
);

template<typename T>
void flash_sdpa_bwd_cudnn(
    const Tile<T> &K,
    const Tile<T> &Q,
    const Tile<T> &V,
    const Tile<T> &O,
    const Tile<T> &dO,
    const Tile<T> &mask,
    const Tile<fp32_t> &logsumexp,
    const Tile<T> &dK,
    const Tile<T> &dQ,
    const Tile<T> &dV
);

} // namespace nntile::tile
