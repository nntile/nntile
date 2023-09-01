/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/prod_fiber3.hh
 * Tile wrappers for per-element product of a tensor and a broadcasted fiber
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

// Tile<T> per-element multiplication of a tensor and a broadcasted fiber
template<typename T>
void prod_fiber3_async(const Tile<T> &src1, T alpha, const Tile<T> &src2,
        const Tile<T> &dst, Index axis);

// Tile<T> per-element multiplication of a tensor and a broadcasted fiber
template<typename T>
void prod_fiber3(const Tile<T> &src1, T alpha, const Tile<T> &src2,
        const Tile<T> &dst, Index axis);

} // namespace tile
} // namespace nntile

