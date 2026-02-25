/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/fill.cc
 * Fill operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/fill.hh"
#include "nntile/tile/fill.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise fill operation
//
// @param[inout] A: Tensor for the element-wise fill operation
template<typename T>
void fill_async(Scalar val, const Tensor<T> &A)
{
    for(Index i = 0; i < A.grid.nelems; ++i)
    {
        auto tile_handle = A.get_tile_handle(i);
        auto tile = A.get_tile(i);
        tile::fill_async<T>(val, tile);
        // Flush cache for the output tile on every node
        tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise fill operation
//
// @param[inout] A: Tensor for the element-wise fill operation
template<typename T>
void fill(Scalar val, const Tensor<T> &A)
{
    fill_async<T>(val, A);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void fill_async<fp32_t>(Scalar val, const Tensor<fp32_t> &A);

template
void fill_async<bf16_t>(Scalar val, const Tensor<bf16_t> &A);

template
void fill_async<fp16_t>(Scalar val, const Tensor<fp16_t> &A);

template
void fill_async<fp32_fast_tf32_t>(Scalar val, const Tensor<fp32_fast_tf32_t> &A);

template
void fill_async<fp32_fast_fp16_t>(Scalar val, const Tensor<fp32_fast_fp16_t> &A);

template
void fill_async<fp32_fast_bf16_t>(Scalar val, const Tensor<fp32_fast_bf16_t> &A);

template
void fill_async<fp64_t>(Scalar val, const Tensor<fp64_t> &A);

// Explicit instantiation
template
void fill<fp32_t>(Scalar val, const Tensor<fp32_t> &A);

template
void fill<bf16_t>(Scalar val, const Tensor<bf16_t> &A);

template
void fill<fp16_t>(Scalar val, const Tensor<fp16_t> &A);

template
void fill<fp32_fast_tf32_t>(Scalar val, const Tensor<fp32_fast_tf32_t> &A);

template
void fill<fp32_fast_fp16_t>(Scalar val, const Tensor<fp32_fast_fp16_t> &A);

template
void fill<fp32_fast_bf16_t>(Scalar val, const Tensor<fp32_fast_bf16_t> &A);

template
void fill<fp64_t>(Scalar val, const Tensor<fp64_t> &A);

} // namespace nntile::tensor
