/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/norm.hh
 * Norm of Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-18
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

template<typename T>
void norm_async(T alpha, const Tile<T> &src, T beta, const Tile<T> &norm_dst,
        Index axis);

template<typename T>
void norm(T alpha, const Tile<T> &src, T beta, const Tile<T> &norm_dst,
        Index axis);

} // namespace tile
} // namespace nntile

