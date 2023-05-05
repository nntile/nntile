/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/nrm2.hh
 * Euclidean norm of Tile<T>
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
void nrm2_async(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst,
        const Tile<T> &tmp);

template<typename T>
void nrm2(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst,
        const Tile<T> &tmp);

} // namespace tile
} // namespace nntile

