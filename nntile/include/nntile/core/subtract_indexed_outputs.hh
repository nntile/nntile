/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/core/subtract_indexed_outputs.hh
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

template<typename T>
void subtract_indexed_outputs_async(int worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<T> &dst, Index ignore_index);

template<typename T>
void subtract_indexed_outputs(int worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<T> &dst, Index ignore_index);

} // namespace nntile::core
