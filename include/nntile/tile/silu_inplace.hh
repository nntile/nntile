/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/silu_inplace.hh
 * Inplace SiLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

// Asynchronous tile-wise SiLU operation
template<typename T>
void silu_inplace_async(const Tile<T> &A);

// Blocking version of tile-wise SiLU operation
template<typename T>
void silu_inplace(const Tile<T> &A);

} // namespace nntile::tile