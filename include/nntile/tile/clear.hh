/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/clear.hh
 * Clear Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-04
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

template<typename T>
void clear_work(const Tile<T> &src);

template<typename T>
void clear_async(const Tile<T> &src)
{
    clear_work<T>(src);
}

template<typename T>
void clear(const Tile<T> &src)
{
    clear_async<T>(src);
    starpu_task_wait_for_all();
}

} // namespace nntile

