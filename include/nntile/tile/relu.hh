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
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

#ifdef NNTILE_USE_CUDA
template<typename T>
void relu_codelet_cuda(void *buffers[], void *cl_args);

extern template
void relu_codelet_cuda<fp32_t>(void *buffers[], void *cl_args);

extern template
void relu_codelet_cuda<fp64_t>(void *buffers[], void *cl_args);
#endif // NNTILE_USE_CUDA

//! Asynchronous tile-wise ReLU operation
//
// @param[inout] A: Tile for the element-wise ReLU operation
template<typename T>
void relu_async(const Tile<T> &A);

extern template
void relu_async(const Tile<fp32_t> &A);

extern template
void relu_async(const Tile<fp64_t> &A);

//! Blocking version of tile-wise ReLU operation
//
// @param[inout] A: Tile for the element-wise ReLU operation
template<typename T>
void relu(const Tile<T> &A)
{
    relu_async<T>(A);
    starpu_task_wait_for_all();
}

} // namespace nntile

