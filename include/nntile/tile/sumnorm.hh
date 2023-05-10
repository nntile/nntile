/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/sumnorm.hh
 * Sum and Euclidean norm of Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-05
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

template<typename T>
void sumnorm_async(const Tile<T> &src, const Tile<T> &dst, Index axis);

template<typename T>
void sumnorm(const Tile<T> &src, const Tile<T> &dst, Index axis);

} // namespace tile
} // namespace nntile

