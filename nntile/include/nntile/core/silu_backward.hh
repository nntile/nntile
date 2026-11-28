/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/core/silu_backward.hh
 * Backward SiLU operation for Tile<T>
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

// Asynchronous tile-wise backward SiLU operation
template<typename T>
void silu_backward_async(int worker_hint, Scalar alpha, const Tile<T> &x, const Tile<T> &dy,
        Scalar beta, const Tile<T> &dx);

// Blocking version of tile-wise backward SiLU operation
template<typename T>
void silu_backward(int worker_hint, Scalar alpha, const Tile<T> &x, const Tile<T> &dy,
        Scalar beta, const Tile<T> &dx);

} // namespace nntile::core
