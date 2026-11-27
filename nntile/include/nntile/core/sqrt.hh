/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/core/sqrt.hh
 * Sqrt operation for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

// Asynchronous tile-wise sqrt operation
template<typename T>
void sqrt_async(int starpu_worker_hint, const Tile<T> &src, const Tile<T> &dst);

// Blocking version of tile-wise sqrt operation
template<typename T>
void sqrt(int starpu_worker_hint, const Tile<T> &src, const Tile<T> &dst);

} // namespace nntile::core
