/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/fill.hh
 * Fill operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-24
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

// Asynchronous tile-wise fill operation
template<typename T>
void fill_async(T val, const Tile<T> &A);

// Blocking version of tile-wise fill operation
template<typename T>
void fill(T val, const Tile<T> &A);

} // namespace tile
} // namespace nntile

