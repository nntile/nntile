/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/sumprod_slice.hh
 * Scalar product of slices of two Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

template<typename T>
void sumprod_slice_async(Scalar alpha, const Tile<T> &src1, const Tile<T> &src2,
        Scalar beta, const Tile<T> &dst, Index axis);

template<typename T>
void sumprod_slice(Scalar alpha, const Tile<T> &src1, const Tile<T> &src2, Scalar beta,
        const Tile<T> &dst, Index axis);

} // namespace nntile::tile
