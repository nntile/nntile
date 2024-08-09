/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/addcdiv.hh
 * Addcdiv operation for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

// Asynchronous tile-wise addcdiv operation
template<typename T>
void addcdiv_async(Scalar val, Scalar eps, const Tile<T> &nom, const Tile<T> &denom, const Tile<T> &src);

// Blocking version of tile-wise addcdiv operation
template<typename T>
void addcdiv(Scalar val, Scalar eps, const Tile<T> &nom, const Tile<T> &denom, const Tile<T> &src);

} // namespace nntile::tile
