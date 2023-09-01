/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/add_slice3.hh
 * Tile wrappers for addition of a tensor and a broadcasted slice
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-03
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

// Tile<T> addition of a tensor and a broadcasted slice
template<typename T>
void add_slice3_async(T alpha, const Tile<T> &src1, T beta,
        const Tile<T> &src2, const Tile<T> &dst, Index axis);

// Tile<T> addition of a tensor and a broadcasted slice
template<typename T>
void add_slice3(T alpha, const Tile<T> &src1, T beta, const Tile<T> &src2,
        const Tile<T> &dst, Index axis);

} // namespace tile
} // namespace nntile

