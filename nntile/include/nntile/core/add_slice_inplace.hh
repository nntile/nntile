/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/core/add_slice_inplace.hh
 * Tile wrappers for addition of a tensor and a broadcasted slice
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

// Tile<T> addition of a tensor and a broadcasted slice
template<typename T>
void add_slice_inplace_async(int worker_hint, Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst, Index axis);

// Tile<T> addition of a tensor and a broadcasted slice
template<typename T>
void add_slice_inplace(int worker_hint, Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst, Index axis);

} // namespace nntile::core
