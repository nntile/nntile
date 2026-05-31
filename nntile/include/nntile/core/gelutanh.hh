/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/core/gelutanh.hh
 * Approximate GeLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

template<typename T>
void gelutanh_async(int starpu_worker_hint, const Tile<T> &src, const Tile<T> &dst);

template<typename T>
void gelutanh(int starpu_worker_hint, const Tile<T> &src, const Tile<T> &dst);

} // namespace nntile::core
