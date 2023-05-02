/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/norm_slice.hh
 * Euclidian norms of fibers into a slice of a Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-02
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

template<typename T>
void norm_slice_async(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst,
        Index axis);

template<typename T>
void norm_slice(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst,
        Index axis);

} // namespace tile
} // namespace nntile

