/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/addcdiv.hh
 * Addcdiv operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-14
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

// Asynchronous tile-wise addcdiv operation
template<typename T>
void addcdiv_async(T val, T eps, const Tile<T> &nom, const Tile<T> &denom, const Tile<T> &src);

// Blocking version of tile-wise addcdiv operation
template<typename T>
void addcdiv(T val, T eps, const Tile<T> &nom, const Tile<T> &denom, const Tile<T> &src);

} // namespace tile
} // namespace nntile

