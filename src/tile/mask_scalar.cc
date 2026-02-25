/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/mask_scalar.cc
 * Mask scalar operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/mask_scalar.hh"
#include "nntile/starpu/mask_scalar.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

//! Asynchronous tile-wise mask scalar operation
/*! @param[inout] A: Tile for the element-wise mask scalar operation
 * */
template<typename T>
void mask_scalar_async(const Tile<bool_t> &mask, Scalar val, const Tile<T> &A,
        Index batch_ndim)
{
    Index effective_batch_ndim = batch_ndim;
    if(mask.ndim != A.ndim-effective_batch_ndim)
    {
        if(batch_ndim == 0 && mask.ndim <= A.ndim)
        {
            effective_batch_ndim = A.ndim - mask.ndim;
        }
        else
        {
            throw std::runtime_error("mask.ndim != A.ndim-batch_ndim");
        }
    }
    for(Index i = 0; i < A.ndim-effective_batch_ndim; ++i)
    {
        if(mask.shape[i] != A.shape[i])
        {
            throw std::runtime_error("mask.shape[i] != A.shape[i]");
        }
    }
    int mpi_rank = starpu_mpi_world_rank();
    int a_rank = A.mpi_get_rank();
    mask.mpi_transfer(a_rank, mpi_rank);
    if(mpi_rank != a_rank)
    {
        return;
    }
    // Submit task without any arguments checked
    starpu::mask_scalar.submit<std::tuple<T>>(
            A.matrix_shape[A.ndim-effective_batch_ndim][0],
            A.matrix_shape[A.ndim-effective_batch_ndim][1],
            mask, val, A);
}

//! Blocking version of tile-wise mask scalar operation
/*! @param[inout] A: Tile for the element-wise mask scalar operation
 * */
template<typename T>
void mask_scalar(const Tile<bool_t> &mask, Scalar val, const Tile<T> &A,
        Index batch_ndim)
{
    mask_scalar_async<T>(mask, val, A, batch_ndim);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void mask_scalar_async<fp32_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_t> &A, Index batch_ndim);

template
void mask_scalar_async<fp32_fast_tf32_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_fast_tf32_t> &A, Index batch_ndim);

template
void mask_scalar_async<fp32_fast_fp16_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_fast_fp16_t> &A, Index batch_ndim);

template
void mask_scalar_async<fp32_fast_bf16_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_fast_bf16_t> &A, Index batch_ndim);

template
void mask_scalar_async<fp64_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp64_t> &A, Index batch_ndim);

template
void mask_scalar_async<bf16_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<bf16_t> &A, Index batch_ndim);

template
void mask_scalar_async<fp16_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp16_t> &A, Index batch_ndim);

// Explicit instantiation
template
void mask_scalar<fp32_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_t> &A, Index batch_ndim);

template
void mask_scalar<fp32_fast_tf32_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_fast_tf32_t> &A, Index batch_ndim);

template
void mask_scalar<fp32_fast_fp16_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_fast_fp16_t> &A, Index batch_ndim);

template
void mask_scalar<fp32_fast_bf16_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_fast_bf16_t> &A, Index batch_ndim);

template
void mask_scalar<fp64_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp64_t> &A, Index batch_ndim);

template
void mask_scalar<bf16_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<bf16_t> &A, Index batch_ndim);

template
void mask_scalar<fp16_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp16_t> &A, Index batch_ndim);

} // namespace nntile::tile
