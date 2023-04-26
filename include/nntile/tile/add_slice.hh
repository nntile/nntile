/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/add_slice.hh
 * Bias operation over fibers from a slice of a Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-26
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

// Tile-wise add_slice operation
template<typename T>
void add_slice_async(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst,
        Index axis);

// Tile-wise add_slice operation
template<typename T>
void add_slice(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst,
        Index axis);

} // namespace tile
} // namespace nntile

