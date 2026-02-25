/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/fill.cc
 * Fill operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/fill.hh"
#include "nntile/starpu/fill.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

//! Asynchronous tile-wise fill operation
/*! @param[inout] A: Tile for the element-wise fill operation
 * */
template<typename T>
void fill_async(Scalar val, const Tile<T> &A)
{
    int mpi_rank = starpu_mpi_world_rank();
    int a_rank = A.mpi_get_rank();
    if(mpi_rank == a_rank)
    {
        // Submit task without any arguments checked
        starpu::fill.submit<std::tuple<T>>(A.nelems, val, A);
    }
}

//! Blocking version of tile-wise flll operation
/*! @param[inout] A: Tile for the element-wise fill operation
 * */
template<typename T>
void fill(Scalar val, const Tile<T> &A)
{
    fill_async<T>(val, A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void fill_async<fp32_t>(Scalar val, const Tile<fp32_t> &A);

template
void fill_async<bf16_t>(Scalar val, const Tile<bf16_t> &A);

template
void fill_async<fp16_t>(Scalar val, const Tile<fp16_t> &A);

template
void fill_async<fp32_fast_tf32_t>(Scalar val, const Tile<fp32_fast_tf32_t> &A);

template
void fill_async<fp32_fast_fp16_t>(Scalar val, const Tile<fp32_fast_fp16_t> &A);

template
void fill_async<fp32_fast_bf16_t>(Scalar val, const Tile<fp32_fast_bf16_t> &A);

template
void fill_async<fp64_t>(Scalar val, const Tile<fp64_t> &A);

// Explicit instantiation
template
void fill<fp32_t>(Scalar val, const Tile<fp32_t> &A);

template
void fill<bf16_t>(Scalar val, const Tile<bf16_t> &A);

template
void fill<fp16_t>(Scalar val, const Tile<fp16_t> &A);

template
void fill<fp32_fast_tf32_t>(Scalar val, const Tile<fp32_fast_tf32_t> &A);

template
void fill<fp32_fast_fp16_t>(Scalar val, const Tile<fp32_fast_fp16_t> &A);

template
void fill<fp32_fast_bf16_t>(Scalar val, const Tile<fp32_fast_bf16_t> &A);

template
void fill<fp64_t>(Scalar val, const Tile<fp64_t> &A);

} // namespace nntile::tile
