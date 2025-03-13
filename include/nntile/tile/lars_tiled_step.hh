/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/lars_tiled_step.hh
 * LarsTiled step for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

// Asynchronous tile-wise LarsTiled step operation
template<typename T>
void lars_tiled_step_async(Index num_iter, Index num_steps, Scalar gamma_0, Scalar momentum, Scalar weight_decay,
    Scalar lars_tiled_coefficient, const Tile<T> &grad, const Tile<T> &momentum_buffer, const Tile<T> &p);
                     

// Blocking version of tile-wise LarsTiled step operation
template<typename T>
void lars_tiled_step(Index num_iter, Index num_steps, Scalar gamma_0, Scalar momentum, Scalar weight_decay,
    Scalar lars_tiled_coefficient, const Tile<T> &grad, const Tile<T> &momentum_buffer, const Tile<T> &p);

} // namespace nntile::tile
