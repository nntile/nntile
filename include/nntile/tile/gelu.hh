/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/gelu.hh
 * GeLU operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

//! Asynchronous tile-wise GeLU operation
//
// @param[inout] A: Tile for the element-wise GeLU operation
template<typename T>
void gelu_work(const Tile<T> &A);

extern template
void gelu_work(const Tile<fp32_t> &A);

extern template
void gelu_work(const Tile<fp64_t> &A);

template<typename T>
void gelu_async(const Tile<T> &A)
{
    // No argument checking
    gelu_work<T>(A);
}


//! Blocking version of tile-wise GeLU operation
//
// @param[inout] A: Tile for the element-wise GeLU operation
template<typename T>
void gelu(const Tile<T> &A)
{
    gelu_async<T>(A);
    starpu_task_wait_for_all();
}

} // namespace nntile

