/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/randn.hh
 * Randn operation for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

template<typename T>
void randn_async(const Tile<T> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

template<typename T>
void randn(const Tile<T> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

} // namespace nntile::tile
