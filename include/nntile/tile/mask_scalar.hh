/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/mask_scalar.hh
 * Mask scalar operation for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

// Asynchronous tile-wise mask scalar operation
template<typename T>
void mask_scalar_async(const Tile<bool_t> &mask, Scalar val, const Tile<T> &A);

// Blocking version of tile-wise mask scalar operation
template<typename T>
void mask_scalar(const Tile<bool_t> &mask, Scalar val, const Tile<T> &A);

} // namespace nntile::tile
