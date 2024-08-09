/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/transpose.cc
 * Transpose operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/transpose.hh"
#include "nntile/starpu/transpose.hh"

namespace nntile::tensor
{

//! Tensor-wise transpose operation
template<typename T>
void transpose_async(Scalar alpha, const Tensor<T> &src, const Tensor<T> &dst,
        Index ndim)
{
    // Check dimensions
    if(ndim <= 0 or ndim >= src.ndim)
    {
        throw std::runtime_error("ndim <= 0 or ndim >= src.ndim");
    }
    if(dst.ndim != src.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }
    // Check shapes of tensors
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(src.shape[(i+ndim) % dst.ndim] != dst.shape[i])
        {
            throw std::runtime_error("src.shape[(i+ndim) % dst.ndim] != "
                    "dst.shape[i]");
        }
        if(src.basetile_shape[(i+ndim) % dst.ndim] != dst.basetile_shape[i])
        {
            throw std::runtime_error("src.basetile_shape[(i+ndim) % dst.ndim] "
                    "!= dst.basetile_shape[i]");
        }
    }
    // Apply per-tile transpose asynchronously as needed
    int mpi_rank = starpu_mpi_world_rank();
    Index grid_m = src.grid.matrix_shape[ndim][0];
    Index grid_n = src.grid.matrix_shape[ndim][1];
    for(Index i = 0; i < grid_m; ++i)
    {
        for(Index j = 0; j < grid_n; ++j)
        {
            // Get handle for corresponding tiles of src and dst
            auto src_tile_handle = src.get_tile_handle(i+j*grid_m);
            auto dst_tile_handle = dst.get_tile_handle(i*grid_n+j);
            // MPI rank of the destination tile
            int dst_tile_rank = dst_tile_handle.mpi_get_rank();
            // Transfer data
            src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
            // Execute only on destination node
            if(mpi_rank == dst_tile_rank)
            {
                auto traits = src.get_tile_traits(i+j*grid_m);
                starpu::transpose::submit<T>(traits.matrix_shape[ndim][0],
                        traits.matrix_shape[ndim][1], alpha, src_tile_handle,
                        dst_tile_handle);
            }
            // Flush cache for the output tile on every node
            dst_tile_handle.mpi_flush();
        }
    }
}

//! Tensor-wise transpose operation
template<typename T>
void transpose(Scalar alpha, const Tensor<T> &src, const Tensor<T> &dst, Index ndim)
{
    transpose_async<T>(alpha, src, dst, ndim);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void transpose_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst, Index ndim);

template
void transpose_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst, Index ndim);

template
void transpose_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst, Index ndim);

template
void transpose_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst, Index ndim);

// Explicit instantiation of template
template
void transpose<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst, Index ndim);

template
void transpose<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst, Index ndim);

template
void transpose<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst, Index ndim);

template
void transpose<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst, Index ndim);

} // namespace nntile::tensor
