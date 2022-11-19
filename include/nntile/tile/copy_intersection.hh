/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/copy_intersection.hh
 * Copy intersection of 2 tiles from one into another
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-19
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

template<typename T>
void copy_intersection_async(const Tile<T> &src,
        const std::vector<Index> &src_offset, const Tile<T> &dst,
        const std::vector<Index> &dst_offset, const Tile<Index> &scratch);

template<typename T>
void copy_intersection(const Tile<T> &src,
        const std::vector<Index> &src_offset, const Tile<T> &dst,
        const std::vector<Index> &dst_offset, const Tile<Index> &scratch);

} // namespace tile
} // namespace nntile

