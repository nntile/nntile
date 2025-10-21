/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/scale_fiber.hh
 * Tile wrappers for scaling of a tensor with a broadcasted fiber
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

// Tile<T> scaling of a tensor with a broadcasted fiber
template<typename T>
void scale_fiber_async(Scalar alpha, const Tile<T> &src, const Tile<T> &dst,
        Index axis, Index batch_ndim);

// Tile<T> scaling of a tensor with a broadcasted fiber
template<typename T>
void scale_fiber(Scalar alpha, const Tile<T> &src, const Tile<T> &dst,
        Index axis, Index batch_ndim);

} // namespace nntile::tile
