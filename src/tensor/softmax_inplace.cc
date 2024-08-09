/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/softmax_inplace.cc
 * softmax_inplace operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/softmax_inplace.hh"
#include "nntile/starpu/softmax_inplace.hh"

namespace nntile::tensor
{

template<typename T>
void softmax_inplace_async(const Tensor<T> &maxsumexp, Scalar alpha,
        const Tensor<T> &dst, Index axis)
{
    // Check inputs
    if(maxsumexp.ndim != dst.ndim)
    {
        throw std::runtime_error("maxsumexp.ndim != dst.ndim");
    }
    // Input shape dimension shall be at least 1
    if(maxsumexp.ndim == 0)
    {
        throw std::runtime_error("maxsumexp.ndim == 0");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= dst.ndim)
    {
        throw std::runtime_error("axis >= dst.ndim");
    }
    // Check shapes
    if(maxsumexp.shape[0] != 2)
    {
        throw std::runtime_error("maxsumexp.shape[0] != 2");
    }
    if(maxsumexp.basetile_shape[0] != 2)
    {
        throw std::runtime_error("maxsumexp.basetile_shape[0] != 2");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != maxsumexp.shape[i+1])
        {
            throw std::runtime_error("dst.shape[i] != maxsumexp.shape[i+1]");
        }
        if(dst.basetile_shape[i] != maxsumexp.basetile_shape[i+1])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "maxsumexp.basetile_shape[i+1]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != maxsumexp.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != maxsumexp.shape[i]");
        }
        if(dst.basetile_shape[i] != maxsumexp.basetile_shape[i])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "maxsumexp.basetile_shape[i]");
        }
    }
    // Prepare
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_size = starpu_mpi_world_size();
    int ret;
    // Apply per-tile softmax_inplace asynchronously as needed
    for(Index i = 0; i < maxsumexp.grid.nelems; ++i)
    {
        // Index of current source tile
        auto maxsumexp_tile_index = maxsumexp.grid.linear_to_index(i);
        // Source tile traits
        auto maxsumexp_tile_traits = maxsumexp.get_tile_traits(i);
        // Source tile handle
        auto maxsumexp_tile_handle = maxsumexp.get_tile_handle(i);
        // Set fixed indices of current destination tile
        std::vector<Index> dst_tile_index(dst.ndim);
        for(Index j = 0; j < axis; ++j)
        {
            dst_tile_index[j] = maxsumexp_tile_index[j+1];
        }
        for(Index j = axis+1; j < dst.ndim; ++j)
        {
            dst_tile_index[j] = maxsumexp_tile_index[j];
        }
        // Loop through all necessary destination tiles
        for(Index j = 0; j < dst.grid.shape[axis]; ++j)
        {
            // Set floating axis
            dst_tile_index[axis] = j;
            // Get linear offset from index
            Index dst_tile_offset = dst.grid.index_to_linear(dst_tile_index);
            // Get destination tile handle
            auto dst_tile_handle = dst.get_tile_handle(dst_tile_offset);
            // MPI rank of the destination tile
            int dst_tile_rank = dst_tile_handle.mpi_get_rank();
            // Transfer data
            maxsumexp_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
            // Execute on destination node
            if(mpi_rank == dst_tile_rank)
            {
                // Get destination tile traits
                auto dst_tile_traits = dst.get_tile_traits(dst_tile_offset);
                // Reshape inputs for simplicity:
                //      maxsumexp -> (2,m,n), dst -> (m,k,n)
                Index m, n, k;
                if(axis == 0)
                {
                    m = 1;
                    // 2 elements per single n
                    n = maxsumexp_tile_traits.nelems / 2;
                    k = dst_tile_traits.shape[0];
                }
                else if(axis == dst.ndim-1)
                {
                    // 2 elements per single m
                    m = maxsumexp_tile_traits.nelems / 2;
                    n = 1;
                    k = dst_tile_traits.shape[axis];
                }
                else
                {
                    m = dst_tile_traits.stride[axis];
                    n = dst_tile_traits.matrix_shape[axis+1][1];
                    k = dst_tile_traits.shape[axis];
                }
                // Insert corresponding task
                starpu::softmax_inplace::submit<T>(m, n, k,
                        maxsumexp_tile_handle, alpha, dst_tile_handle);
            }
            // Flush cache for the output tile on every node
            dst_tile_handle.mpi_flush();
        }
    }
}

template<typename T>
void softmax_inplace(const Tensor<T> &maxsumexp, Scalar alpha, const Tensor<T> &dst,
        Index axis)
{
    softmax_inplace_async<T>(maxsumexp, alpha, dst, axis);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void softmax_inplace_async<fp32_t>(const Tensor<fp32_t> &maxsumexp,
        Scalar alpha, const Tensor<fp32_t> &dst, Index axis);

template
void softmax_inplace_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &maxsumexp,
        Scalar alpha, const Tensor<fp32_fast_tf32_t> &dst, Index axis);

template
void softmax_inplace_async<fp64_t>(const Tensor<fp64_t> &maxsumexp,
        Scalar alpha, const Tensor<fp64_t> &dst, Index axis);

template
void softmax_inplace_async<bf16_t>(const Tensor<bf16_t> &maxsumexp, Scalar alpha,
        const Tensor<bf16_t> &dst, Index axis);

// Explicit instantiation
template
void softmax_inplace<fp32_t>(const Tensor<fp32_t> &maxsumexp, Scalar alpha,
        const Tensor<fp32_t> &dst, Index axis);

template
void softmax_inplace<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &maxsumexp, Scalar alpha,
        const Tensor<fp32_fast_tf32_t> &dst, Index axis);

template
void softmax_inplace<fp64_t>(const Tensor<fp64_t> &maxsumexp, Scalar alpha,
        const Tensor<fp64_t> &dst, Index axis);

template
void softmax_inplace<bf16_t>(const Tensor<bf16_t> &maxsumexp, Scalar alpha,
        const Tensor<bf16_t> &dst, Index axis);

} // namespace nntile::tensor
