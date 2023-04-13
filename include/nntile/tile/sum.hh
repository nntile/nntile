/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/sum.hh
 * Sum  of Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Konstantin Sozykin
 * @date 2023-04-13
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

template<typename T>
void sum_async(T alpha, const Tile<T> &src, T beta, const Tile<T> &sum_dst,
        Index axis);

template<typename T>
void sum(T alpha, const Tile<T> &src, T beta, const Tile<T> &sum_dst,
        Index axis);

} // namespace tile
} // namespace nntile

