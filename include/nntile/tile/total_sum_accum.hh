/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/total_sum_accum.hh
 * Total sum accumulating for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-11-11
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

template<typename T>
void total_sum_accum_async(T alpha, const Tile<T> &logsumexp, const Tile<T> &src,
                           const Tile<Index> &class_labels, const Tile<T> &val);

template<typename T>
void total_sum_accum(T alpha, const Tile<T> &logsumexp, const Tile<T> &src,
                     const Tile<Index> &class_labels, const Tile<T> &val);

} // namespace tile
} // namespace nntile
