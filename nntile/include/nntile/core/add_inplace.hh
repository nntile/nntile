/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/core/add_inplace.hh
 * Add operation for Tile<T>'s
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

// Tile-wise add operation
template<typename T>
void add_inplace_async(int worker_hint, Scalar alpha, const Tile<T> &src, Scalar beta,
        const Tile<T> &dst);

// Tile-wise add operation
template<typename T>
void add_inplace(int worker_hint, Scalar alpha, const Tile<T> &src, Scalar beta,
        const Tile<T> &dst);

} // namespace nntile::core
