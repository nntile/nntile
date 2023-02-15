/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/sumnorm.hh
 * Sum and Euclidian norm of Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-31
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

template<typename T>
void sum_async(const Tile<T> &src, const Tile<T> &dst, Index axis);

template<typename T>
void sum(const Tile<T> &src, const Tile<T> &dst, Index axis);

} // namespace tile
} // namespace nntile

