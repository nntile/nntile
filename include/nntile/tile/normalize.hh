/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/normalize.hh
 * Normalize operation for Tile<T>
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

template<typename T>
void normalize_async(const Tile<T> &gamma_beta, const Tile<T> &sumnorm,
        const Tile<T> &dst, Index size, Scalar eps, Index axis);

template<typename T>
void normalize(const Tile<T> &gamma_beta, const Tile<T> &sumnorm,
        const Tile<T> &dst, Index size, Scalar eps, Index axis);

} // namespace nntile::tile
