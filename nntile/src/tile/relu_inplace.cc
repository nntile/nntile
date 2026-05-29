/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/relu_inplace.cc
 * Inplace ReLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/relu_inplace.hh"
#include "nntile/starpu/relu_inplace.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

//! Asynchronous tile-wise ReLU operation
/*! @param[inout] A: Tile for the element-wise ReLU operation
 * */
template<typename T>
void relu_inplace_async(const Tile<T> &A)
{
    int mpi_rank = starpu_mpi_world_rank();
    int a_rank = A.mpi_get_rank();
    if(mpi_rank == a_rank)
    {
        // Submit task without any arguments checked
        starpu::relu_inplace.submit<std::tuple<T>>(A.nelems, A);
    }
}

//! Blocking version of tile-wise ReLU operation
/*! @param[inout] A: Tile for the element-wise ReLU operation
 * */
template<typename T>
void relu_inplace(const Tile<T> &A)
{
    relu_inplace_async<T>(A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void relu_inplace_async<fp32_t>(const Tile<fp32_t> &A);

template
void relu_inplace_async<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &A);

template
void relu_inplace_async<fp32_fast_fp16_t>(const Tile<fp32_fast_fp16_t> &A);

template
void relu_inplace_async<fp32_fast_bf16_t>(const Tile<fp32_fast_bf16_t> &A);

template
void relu_inplace_async<bf16_t>(const Tile<bf16_t> &A);

template
void relu_inplace_async<fp16_t>(const Tile<fp16_t> &A);

template
void relu_inplace_async<fp64_t>(const Tile<fp64_t> &A);

// Explicit instantiation
template
void relu_inplace<fp32_t>(const Tile<fp32_t> &A);

template
void relu_inplace<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &A);

template
void relu_inplace<fp32_fast_fp16_t>(const Tile<fp32_fast_fp16_t> &A);

template
void relu_inplace<fp32_fast_bf16_t>(const Tile<fp32_fast_bf16_t> &A);

template
void relu_inplace<bf16_t>(const Tile<bf16_t> &A);

template
void relu_inplace<fp16_t>(const Tile<fp16_t> &A);

template
void relu_inplace<fp64_t>(const Tile<fp64_t> &A);

} // namespace nntile::tile
