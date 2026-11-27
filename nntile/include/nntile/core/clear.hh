/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/core/clear.hh
 * Clear Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

// Clear a tile
template<typename T>
void clear_async(int worker_hint, const Tile<T> &tile);

// Clear a tile
template<typename T>
void clear(int worker_hint, const Tile<T> &tile);

} // namespace nntile::core
