/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/multiply.hh
 * Per-element product of two Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

// Asynchronous tile-wise multiply operation
template<typename T>
void multiply_async(Scalar alpha, const Tile<T> &src1, const Tile<T> &src2,
        const Tile<T> &dst);

// Blocking version of tile-wise multiply operation
template<typename T>
void multiply(Scalar alpha, const Tile<T> &src1, const Tile<T> &src2,
        const Tile<T> &dst);

} // namespace nntile::tile
