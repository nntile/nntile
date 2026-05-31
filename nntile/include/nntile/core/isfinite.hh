/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/core/isfinite.hh
 * Check NaN or Inf elements for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

// Asynchronous tile-wise check Inf/NaN operation
template<typename T>
void isfinite_async(int starpu_worker_hint, const Tile<T> &A, const Tile<bool_t> &flag);

// Blocking version of tile-wise check Inf/NaN operation
template<typename T>
void isfinite(int starpu_worker_hint, const Tile<T> &A, const Tile<bool_t> &flag);

} // namespace nntile::core
