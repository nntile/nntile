/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/biasprod_outer.hh
 * Bias-like product along outer axes operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-20
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

// Tile-wise biasprod_outer operation
template<typename T>
void biasprod_outer_async(const Tile<T> &src, const Tile<T> &dst, Index axis);

// Tile-wise biasprod_outer operation
template<typename T>
void biasprod_outer(const Tile<T> &src, const Tile<T> &dst, Index axis);

} // namespace tile
} // namespace nntile

