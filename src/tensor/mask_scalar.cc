/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/mask_scalar.cc
 * Mask scalar operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/mask_scalar.hh"
#include "nntile/starpu/mask_scalar.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise mask calar operation
//
// @param[inout] A: Tensor for the element-wise fill operation
template<typename T>
void mask_scalar_async(const Tensor<bool_t> &mask, Scalar val, const Tensor<T> &A,
        Index batch_ndim)
{
    if(mask.ndim != A.ndim-batch_ndim)
    {
        throw std::runtime_error("mask.ndim != A.ndim-batch_ndim");
    }
    // Check shapes and tiles
    for(Index i = 0; i < A.ndim-batch_ndim; ++i)
    {
        if(A.shape[i] != mask.shape[i])
        {
            throw std::runtime_error("A.shape[i] != mask.shape[i]");
        }
        if(A.basetile_shape[i] != mask.basetile_shape[i])
        {
            throw std::runtime_error("A.basetile_shape[i] != "
                    "mask.basetile_shape[i]");
        }
    }
    // Run the code
    int mpi_rank = starpu_mpi_world_rank();
    for(Index i = 0; i < A.grid.nelems; ++i)
    {
        auto A_tile_handle = A.get_tile_handle(i);
        auto A_tile_index = A.grid.linear_to_index(i);
        int A_tile_rank = A_tile_handle.mpi_get_rank();
        std::vector<Index> mask_tile_index(mask.ndim);
        for(Index j = 0; j < mask.ndim; ++j)
        {
            mask_tile_index[j] = A_tile_index[j];
        }
        auto mask_tile_handle = mask.get_tile_handle(mask_tile_index);
        int mask_tile_rank = mask_tile_handle.mpi_get_rank();
        mask_tile_handle.mpi_transfer(A_tile_rank, mpi_rank);
        // Execute only on node-owner
        if(mpi_rank == A_tile_rank)
        {
            auto tile_traits = A.get_tile_traits(i);
            starpu::mask_scalar::submit<T>(
                    tile_traits.matrix_shape[A.ndim-batch_ndim][0], \
                    tile_traits.matrix_shape[A.ndim-batch_ndim][1], \
                    mask_tile_handle, val, A_tile_handle);
        }
        // Flush cache for the output tile on every node
        A_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise mask scalar operation
//
// @param[inout] A: Tensor for the element-wise mask scalar operation
template<typename T>
void mask_scalar(const Tensor<bool_t> &mask, Scalar val, const Tensor<T> &A,
        Index batch_ndim)
{
    mask_scalar_async<T>(mask, val, A, batch_ndim);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void mask_scalar_async<fp32_t>(const Tensor<bool_t> &mask, Scalar val,
        const Tensor<fp32_t> &A, Index batch_ndim);

template
void mask_scalar_async<fp32_fast_tf32_t>(const Tensor<bool_t> &mask, Scalar val,
        const Tensor<fp32_fast_tf32_t> &A, Index batch_ndim);

template
void mask_scalar_async<fp64_t>(const Tensor<bool_t> &mask, Scalar val,
        const Tensor<fp64_t> &A, Index batch_ndim);

template
void mask_scalar_async<bf16_t>(const Tensor<bool_t> &mask, Scalar val,
        const Tensor<bf16_t> &A, Index batch_ndim);

// Explicit instantiation
template
void mask_scalar<fp32_t>(const Tensor<bool_t> &mask, Scalar val,
        const Tensor<fp32_t> &A, Index batch_ndim);

template
void mask_scalar<fp32_fast_tf32_t>(const Tensor<bool_t> &mask, Scalar val,
        const Tensor<fp32_fast_tf32_t> &A, Index batch_ndim);

template
void mask_scalar<fp64_t>(const Tensor<bool_t> &mask, Scalar val,
        const Tensor<fp64_t> &A, Index batch_ndim);

template
void mask_scalar<bf16_t>(const Tensor<bool_t> &mask, Scalar val,
        const Tensor<bf16_t> &A, Index batch_ndim);

} // namespace nntile::tensor
