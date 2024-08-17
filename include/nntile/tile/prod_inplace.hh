/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/prod_inplace.hh
 * Per-element product of two Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

template<typename T>
void prod_inplace_async(const Tile<T> &src, const Tile<T> &dst);

template<typename T>
void prod_inplace(const Tile<T> &src, const Tile<T> &dst);

} // namespace nntile::tile
