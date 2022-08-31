/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/copy.hh
 * Copy operation for Tile<T>
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
void copy_async(const Tile<T> &src, const std::vector<Index> &src_offset,
        const Tile<T> &dst, const std::vector<Index> &dst_offset);

template<typename T>
void copy(const Tile<T> &src, const std::vector<Index> &src_offset,
        const Tile<T> &dst, const std::vector<Index> &dst_offset);

} // namespace tile
} // namespace nntile

