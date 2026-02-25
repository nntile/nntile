/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/transpose.hh
 * Transpose operation for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

// Tile-wise transpose operation
template<typename T>
void transpose_async(Scalar alpha, const Tile<T> &src, const Tile<T> &dst,
        Index ndim);

// Tile-wise transpose operation
template<typename T>
void transpose(Scalar alpha, const Tile<T> &src, const Tile<T> &dst,
        Index ndim);

} // namespace nntile::tile
