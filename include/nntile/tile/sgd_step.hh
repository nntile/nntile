/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/sgd_step.hh
 * SGD with momentum step for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

// Asynchronous tile-wise SGD with momentum step operation
template<typename T>
void sgd_step_async(Scalar momentum, Scalar lr, Scalar weight_decay,
                     const Tile<T> &grad, const Tile<T> &velocity,
                     const Tile<T> &p);

// Blocking version of tile-wise SGD with momentum step operation
template<typename T>
void sgd_step(Scalar momentum, Scalar lr, Scalar weight_decay,
               const Tile<T> &grad, const Tile<T> &velocity,
               const Tile<T> &p);

} // namespace nntile::tile
