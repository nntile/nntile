/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/logsumexp.hh
 * Logsumexp operation of Tile<T> based on the result of maxsumexp operation
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-15
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

template<typename T>
void logsumexp_async(const Tile<T> &src, const Tile<T> &dst);

template<typename T>
void logsumexp(const Tile<T> &src, const Tile<T> &dst);

} // namespace tile
} // namespace nntile

