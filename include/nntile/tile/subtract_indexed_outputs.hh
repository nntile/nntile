/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/subtract_indexed_outputs.hh
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

template<typename T>
void subtract_indexed_outputs_async(Scalar val, const Tile<int64_t> &labels,
        const Tile<T> &dst);

template<typename T>
void subtract_indexed_outputs(Scalar val, const Tile<int64_t> &labels,
        const Tile<T> &dst);

} // namespace nntile::tile
