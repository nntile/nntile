/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/relu.hh
 * ReLU operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-31
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

// Asynchronous tile-wise ReLU operation
template<typename T>
void relu_async(const Tile<T> &A);

// Blocking version of tile-wise ReLU operation
template<typename T>
void relu(const Tile<T> &A);

} // namespace tile
} // namespace nntile

