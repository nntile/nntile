/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/core/relu_inplace.hh
 * Inplace ReLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

// Asynchronous tile-wise ReLU operation
template<typename T>
void relu_inplace_async(const Tile<T> &A);

// Blocking version of tile-wise ReLU operation
template<typename T>
void relu_inplace(const Tile<T> &A);

} // namespace nntile::core
