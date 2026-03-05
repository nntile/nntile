/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/rope_backward.hh
 * Backward RoPE operation for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

template<typename T>
void rope_backward_async(const Tile<T> &sin, const Tile<T> &cos,
        const Tile<T> &dy, const Tile<T> &dx);

template<typename T>
void rope_backward(const Tile<T> &sin, const Tile<T> &cos,
        const Tile<T> &dy, const Tile<T> &dx);

} // namespace nntile::tile
