/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/hypot.hh
 * hypot operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-03
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

// Tile-wise hypot operation
template<typename T>
void hypot_async(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst);

// Tile-wise hypot operation
template<typename T>
void hypot(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst);

} // namespace tile
} // namespace nntile

