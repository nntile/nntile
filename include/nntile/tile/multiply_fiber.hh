/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/prod_fiber.hh
 * Tile wrappers for per-element product of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

// Tile<T> per-element multiplication of a tensor and a broadcasted fiber
template<typename T>
void prod_fiber_async(const Tile<T> &src, Scalar alpha, const Tile<T> &dst,
        Index axis);

// Tile<T> per-element multiplication of a tensor and a broadcasted fiber
template<typename T>
void prod_fiber(const Tile<T> &src, Scalar alpha, const Tile<T> &dst, Index axis);

} // namespace nntile::tile
