/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/total_sum_accum.hh
 * Total sum accumulating for Tile<T>
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

template<typename T>
void total_sum_accum_async(scal_t alpha, const Tile<T> &logsumexp, const Tile<T> &src,
                           const Tile<Index> &class_labels, const Tile<T> &val);

template<typename T>
void total_sum_accum(scal_t alpha, const Tile<T> &logsumexp, const Tile<T> &src,
                     const Tile<Index> &class_labels, const Tile<T> &val);

} // namespace nntile::tile
