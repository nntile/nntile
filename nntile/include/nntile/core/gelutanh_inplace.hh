/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/core/gelutanh_inplace.hh
 * Approximate GeLU operation for Tile<T>
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

template<typename T>
void gelutanh_inplace_async(int worker_hint, const Tile<T> &A);

template<typename T>
void gelutanh_inplace(int worker_hint, const Tile<T> &A);

} // namespace nntile::core
