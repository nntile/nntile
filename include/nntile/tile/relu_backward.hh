/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/relu_backward.hh
 * Backward ReLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

// Asynchronous tile-wise backward ReLU operation
template<typename T>
void relu_backward_async(const Tile<T> &x, const Tile<T> &dy,
        const Tile<T> &dx);

// Blocking version of tile-wise backward ReLU operation
template<typename T>
void relu_backward(const Tile<T> &x, const Tile<T> &dy, const Tile<T> &dx);

} // namespace nntile::tile
