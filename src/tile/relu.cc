/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/relu.cc
 * ReLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/relu.hh"
#include "nntile/starpu/relu.hh"

namespace nntile::tile
{

//! Asynchronous tile-wise ReLU operation
/*! @param[inout] A: Tile for the element-wise ReLU operation
 * */
template<typename T>
void relu_async(const Tile<T> &A)
{
    // Submit task without any arguments checked
    starpu::relu::submit<T>(A.nelems, A);
}

//! Blocking version of tile-wise ReLU operation
/*! @param[inout] A: Tile for the element-wise ReLU operation
 * */
template<typename T>
void relu(const Tile<T> &A)
{
    relu_async<T>(A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void relu_async<fp32_t>(const Tile<fp32_t> &A);

template
void relu_async<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &A);

template
void relu_async<fp64_t>(const Tile<fp64_t> &A);

// Explicit instantiation
template
void relu<fp32_t>(const Tile<fp32_t> &A);

template
void relu<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &A);

template
void relu<fp64_t>(const Tile<fp64_t> &A);

} // namespace nntile::tile
