/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/mask_scalar.hh
 * Mask scalar operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-22
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

// Asynchronous tile-wise mask scalar operation
template<typename T>
void mask_scalar_async(const Tile<bool_t> &mask, T val, const Tile<T> &A);

// Blocking version of tile-wise mask scalar operation
template<typename T>
void mask_scalar(const Tile<bool_t> &mask, T val, const Tile<T> &A);

} // namespace tile
} // namespace nntile