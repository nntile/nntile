/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/core/rope_backward.hh
 * Backward RoPE operation for Tile<T>
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

template<typename T>
void rope_backward_async(int worker_hint, const Tile<T> &sin, const Tile<T> &cos,
        const Tile<T> &dy, const Tile<T> &dx, Index sin_pair0 = 0);

template<typename T>
void rope_backward(int worker_hint, const Tile<T> &sin, const Tile<T> &cos,
        const Tile<T> &dy, const Tile<T> &dx, Index sin_pair0 = 0);

} // namespace nntile::core
