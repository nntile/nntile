/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/core/rope.hh
 * Tile wrappers for the Rotary Positional Embedding
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

// Tile<T> RoPE
template<typename T>
void rope_async(int starpu_worker_hint, const Tile<T> &sin, const Tile<T> &cos,
        const Tile<T> &src, const Tile<T> &dst, Index sin_pair0 = 0);

// Tile<T> RoPE
template<typename T>
void rope(int starpu_worker_hint, const Tile<T> &sin, const Tile<T> &cos,
        const Tile<T> &src, const Tile<T> &dst, Index sin_pair0 = 0);

} // namespace nntile::core
