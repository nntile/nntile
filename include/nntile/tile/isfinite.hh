/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/isfinite.hh
 * Check NaN or Inf elements for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

// Asynchronous tile-wise power operation
template<typename T>
void isfinite_async(const Tile<T> &A, const Tile<bool_t> &flag);

// Blocking version of tile-wise power operation
template<typename T>
void isfinite(const Tile<T> &A, const Tile<bool_t> &flag);

} // namespace nntile::tile
