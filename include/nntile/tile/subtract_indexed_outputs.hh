/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/subtract_indexed_outputs.hh
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-28
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

template<typename T>
void subtract_indexed_outputs_async(T val, const Tile<Index> &labels,
        const Tile<T> &dst);

template<typename T>
void subtract_indexed_outputs(T val, const Tile<Index> &labels,
        const Tile<T> &dst);

} // namespace tile
} // namespace nntile
