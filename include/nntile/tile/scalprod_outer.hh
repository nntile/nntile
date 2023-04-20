/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/scalprod_outer.hh
 * Scalar products of two Tile<T> along outer axes
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-20
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

template<typename T>
void scalprod_outer_async(T alpha, const Tile<T> &src1, const Tile<T> &src2,
        T beta, const Tile<T> &dst, Index axis);

template<typename T>
void scalprod_outer(T alpha, const Tile<T> &src1, const Tile<T> &src2, T beta,
        const Tile<T> &dst, Index axis);

} // namespace tile
} // namespace nntile

