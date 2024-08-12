/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/add_slice3.hh
 * Tile wrappers for the Rotary Positional Embedding
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

// Tile<T> RoPE
template<typename T>
void rope_async(const Tile<T> &sin, const Tile<T> &cos,
        const Tile<T> &src, const Tile<T> &dst, Index axis);

// Tile<T> RoPE
template<typename T>
void rope(const Tile<T> &sin, const Tile<T> &cos,
        const Tile<T> &src, const Tile<T> &dst, Index axis);

} // namespace nntile::tile
