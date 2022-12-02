/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/clear.cc
 * Clear Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-12-02
 * */

#include "nntile/tile/clear.hh"
#include "nntile/starpu/clear.hh"

namespace nntile
{
namespace tile
{

//! Asynchronously clear a tile
template<typename T>
void clear_async(const Tile<T> &tile)
{
    starpu::clear::submit(tile);
}

//! Asynchronously clear a tile
template<typename T>
void clear(const Tile<T> &tile)
{
    clear_async<T>(tile);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void clear_async<fp32_t>(const Tile<fp32_t> &tile);

template
void clear_async<fp64_t>(const Tile<fp64_t> &tile);

// Explicit instantiation
template
void clear<fp32_t>(const Tile<fp32_t> &tile);

template
void clear<fp64_t>(const Tile<fp64_t> &tile);

} // namespace tile
} // namespace nntile

