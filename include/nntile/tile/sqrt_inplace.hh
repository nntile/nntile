/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/sqrt_inplace.hh
 * Inplace sqrt operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-07-01
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

// Asynchronous tile-wise sqrt operation
template<typename T>
void sqrt_inplace_async(const Tile<T> &A);

// Blocking version of tile-wise sqrt operation
template<typename T>
void sqrt_inplace(const Tile<T> &A);

} // namespace tile
} // namespace nntile

