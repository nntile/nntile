/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/core/hypot.hh
 * hypot operation for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

// Tile-wise hypot operation
template<typename T>
void hypot_async(int starpu_worker_hint, Scalar alpha, const Tile<T> &src1, Scalar beta, const Tile<T> &src2, const Tile<T> &dst);

// Tile-wise hypot operation
template<typename T>
void hypot(int starpu_worker_hint, Scalar alpha, const Tile<T> &src1, Scalar beta, const Tile<T> &src2, const Tile<T> &dst);

} // namespace nntile::core
