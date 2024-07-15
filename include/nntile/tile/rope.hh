/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/add_slice3.hh
 * Tile wrappers for the Rotary Positional Embedding
 *
 * @version 1.0.0
 * @author Gleb Karpov
 * @date 2024-06-27
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

// Tile<T> addition of a tensor and a broadcasted slice
template<typename T>
void rope_async(const Tile<T> &sin, const Tile<T> &cos, 
        const Tile<T> &src, const Tile<T> &dst, Index axis);

// Tile<T> addition of a tensor and a broadcasted slice
template<typename T>
void rope(const Tile<T> &sin, const Tile<T> &cos, 
        const Tile<T> &src, const Tile<T> &dst, Index axis);

} // namespace tile
} // namespace nntile