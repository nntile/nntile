/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/normalize.hh
 * Normalize operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-02
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

template<typename T>
void normalize_async(const StarpuVariableHandle &gamma_beta,
        const Tile<T> &sumnorm, const Tile<T> &dst, Index l, T eps,
        Index axis);

template<typename T>
void normalize(const StarpuVariableHandle &gamma_beta,
        const Tile<T> &sumnorm, const Tile<T> &dst, Index l, T eps,
        Index axis);

} // namespace tile
} // namespace nntile

