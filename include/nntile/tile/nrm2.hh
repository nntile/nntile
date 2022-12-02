/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/nrm2.hh
 * Euclidian norm of Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-12-02
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

template<typename T>
void nrm2_async(const Tile<T> &src, const Tile<T> &dst);

template<typename T>
void nrm2(const Tile<T> &src, const Tile<T> &dst);

} // namespace tile
} // namespace nntile

