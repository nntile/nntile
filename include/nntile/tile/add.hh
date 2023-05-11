/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/add.hh
 * Add operation for two Tile<T>'s
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-05-08
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

// Tile-wise add operation
template<typename T>
void add_async(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst);

// Tile-wise add operation
template<typename T>
void add(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst);

} // namespace tile
} // namespace nntile

