/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/axpy.hh
 * AXPY for two Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-23
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

template<typename T>
void axpy_async(const Tile<T> &alpha, const Tile<T> &src, const Tile<T> &dst);

template<typename T>
void axpy(const Tile<T> &alpha, const Tile<T> &src, const Tile<T> &dst);

template<typename T>
void axpy2_async(T alpha, const Tile<T> &src, const Tile<T> &dst);

template<typename T>
void axpy2(T alpha, const Tile<T> &src, const Tile<T> &dst);

} // namespace tile
} // namespace nntile

