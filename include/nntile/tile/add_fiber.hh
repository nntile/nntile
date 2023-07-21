/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/add_fiber.hh
 * Tile wrappers for addition of a tensor and a broadcasted fiber
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-21
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

// Tile<T> addition of a tensor and a broadcasted fiber
template<typename T>
void add_fiber_async(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst,
        Index axis, Index batch_ndim);

// Tile<T> addition of a tensor and a broadcasted fiber
template<typename T>
void add_fiber(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst,
        Index axis, Index batch_ndim);

} // namespace tile
} // namespace nntile

