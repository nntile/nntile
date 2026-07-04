/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/core/swap_two_axes.hh
 * swap_two_axes operation for Tile<T>.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/tile.hh>

namespace nntile::core
{

template<typename T>
void swap_two_axes_async(
    int starpu_worker_hint,
    const Tile<T> &src,
    const Tile<T> &dst,
    Index dim0,
    Index dim1);

template<typename T>
void swap_two_axes(
    int starpu_worker_hint,
    const Tile<T> &src,
    const Tile<T> &dst,
    Index dim0,
    Index dim1);

} // namespace nntile::core
