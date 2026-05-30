/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/core/scale_slice.hh
 * Tile wrappers for scaling of a broadcasted slice
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

// Tile<T> scaling of a broadcasted slice
template<typename T>
void scale_slice_async(Scalar alpha, const Tile<T> &src, const Tile<T> &dst, Index axis);

// Tile<T> scaling of a broadcasted slice
template<typename T>
void scale_slice(Scalar alpha, const Tile<T> &src, const Tile<T> &dst, Index axis);

} // namespace nntile::core
