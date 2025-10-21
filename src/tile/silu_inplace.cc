/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/silu_inplace.cc
 * Inplace SiLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/silu_inplace.hh"
#include "nntile/starpu/silu_inplace.hh"

namespace nntile::tile
{

//! Asynchronous tile-wise SiLU operation
/*! @param[inout] A: Tile for the element-wise SiLU operation
 * */
template<typename T>
void silu_inplace_async(const Tile<T> &A)
{
    // Submit task without any arguments checked
    starpu::silu_inplace.submit<std::tuple<T>>(A.nelems, A);
}

//! Blocking version of tile-wise SiLU operation
/*! @param[inout] A: Tile for the element-wise SiLU operation
 * */
template<typename T>
void silu_inplace(const Tile<T> &A)
{
    silu_inplace_async<T>(A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void silu_inplace_async<fp32_t>(const Tile<fp32_t> &A);

template
void silu_inplace_async<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &A);

template
void silu_inplace_async<fp32_fast_fp16_t>(const Tile<fp32_fast_fp16_t> &A);

template
void silu_inplace_async<fp32_fast_bf16_t>(const Tile<fp32_fast_bf16_t> &A);

template
void silu_inplace_async<bf16_t>(const Tile<bf16_t> &A);

template
void silu_inplace_async<fp16_t>(const Tile<fp16_t> &A);

template
void silu_inplace_async<fp64_t>(const Tile<fp64_t> &A);

// Explicit instantiation
template
void silu_inplace<fp32_t>(const Tile<fp32_t> &A);

template
void silu_inplace<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &A);

template
void silu_inplace<fp32_fast_fp16_t>(const Tile<fp32_fast_fp16_t> &A);

template
void silu_inplace<fp32_fast_bf16_t>(const Tile<fp32_fast_bf16_t> &A);

template
void silu_inplace<bf16_t>(const Tile<bf16_t> &A);

template
void silu_inplace<fp16_t>(const Tile<fp16_t> &A);

template
void silu_inplace<fp64_t>(const Tile<fp64_t> &A);

} // namespace nntile::tile