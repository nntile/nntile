/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/maximum.hh
 * Per-element maximum of two Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-05-05
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

template<typename T>
void maximum_async(const Tile<T> &src, const Tile<T> &dst);

template<typename T>
void maximum(const Tile<T> &src, const Tile<T> &dst);

} // namespace tile
} // namespace nntile

