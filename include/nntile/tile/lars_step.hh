/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/lars_step.hh
 * LARS step for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/norm_traits.hh>
#include <nntile/tile/tile.hh>

namespace nntile::tile
{

// Asynchronous tile-wise LARS step operation
template<typename T>
void lars_step_async(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
                     const Tile<T> &grad, const Tile<T> &p,
                     const Tile<norm_value_t<T>> &grad_norm, const Tile<norm_value_t<T>> &p_norm);

// Blocking version of tile-wise LARS step operation
template<typename T>
void lars_step(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
               const Tile<T> &grad, const Tile<T> &p,
               const Tile<norm_value_t<T>> &grad_norm, const Tile<norm_value_t<T>> &p_norm);

} // namespace nntile::tile
