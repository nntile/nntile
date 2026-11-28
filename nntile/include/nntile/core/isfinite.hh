/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/core/isfinite.hh
 * Check NaN or Inf elements for Tile<T>
 *
 * @version 2.0.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

// Asynchronous tile-wise check Inf/NaN operation
template<typename T>
void isfinite_async(int worker_hint, const Tile<T> &A, const Tile<bool_t> &flag);

// Blocking version of tile-wise check Inf/NaN operation
template<typename T>
void isfinite(int worker_hint, const Tile<T> &A, const Tile<bool_t> &flag);

} // namespace nntile::core
