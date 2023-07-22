/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/adam_step.hh
 * Adam step for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-07-21
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

// Asynchronous tile-wise Adam step operation
template<typename T>
void adam_step_async(Index num_iter, T beta_1, T beta_2, T eps, T lr, T weight_decay,
                     const Tile<T> &grad, const Tile<T> &first_moment, const Tile<T> &second_moment,
                     const Tile<T> &p);

// Blocking version of tile-wise Adam step operation
template<typename T>
void adam_step(Index num_iter, T beta_1, T beta_2, T eps, T lr, T weight_decay,
               const Tile<T> &grad, const Tile<T> &first_moment, const Tile<T> &second_moment,
               const Tile<T> &p);

} // namespace tile
} // namespace nntile

