/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/core/norm_fiber.hh
 * Euclidean norms over slices into a fiber of a Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

// Tile-wise norm_fiber
template<typename T>
void norm_fiber_async(int starpu_worker_hint, Scalar alpha, const Tile<T> &src1, Scalar beta, const Tile<T> &src2, const Tile<T> &dst,
        Index axis, Index batch_ndim, int redux=0);

// Tile-wise norm_fiber
template<typename T>
void norm_fiber(int starpu_worker_hint, Scalar alpha, const Tile<T> &src1, Scalar beta, const Tile<T> &src2, const Tile<T> &dst,
        Index axis, Index batch_ndim, int redux=0);

} // namespace nntile::core
