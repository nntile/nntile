/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/bias.cc
 * Bias operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-12
 * */

#include "nntile/tensor/bias.hh"
#include "nntile/starpu/bias.hh"

namespace nntile
{
namespace tensor
{

//! Tensor-wise bias operation
template<typename T>
void bias_async(const Tensor<T> &src, const Tensor<T> &dst, Index axis)
{
    // Check dimensions
    if(dst.ndim != src.ndim+1)
    {
        throw std::runtime_error("dst.ndim != src.ndim+1");
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
    // Check shapes of tensors
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
        if(dst.basetile_shape[i] != src.basetile_shape[i])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "src.basetile_shape[i]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i-1])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i-1]");
        }
        if(dst.basetile_shape[i] != src.basetile_shape[i-1])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "src.basetile_shape[i-1]");
        }
    }
    // Apply per-tile bias asynchronously as needed
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        // Index of current source tile
        auto src_tile_index = src.grid.linear_to_index(i);
        // Source tile traits
        auto src_tile_traits = src.get_tile_traits(i);
        // Source tile handle
        auto src_tile_handle = src.get_tile_handle(i);
        // MPI rank and tag of the source tile
        int src_tile_rank = starpu_mpi_data_get_rank(src_tile_handle);
        auto tile_tag = starpu_mpi_data_get_tag(src_tile_handle);
        // Set fixed indices of current destination tile
        std::vector<Index> dst_tile_index(dst.ndim);
        for(Index j = 0; j < axis; ++j)
        {
            dst_tile_index[j] = src_tile_index[j];
        }
        for(Index j = axis+1; j < dst.ndim; ++j)
        {
            dst_tile_index[j] = src_tile_index[j-1];
        }
        // Loop through all necessary destination tiles
        for(Index j = 0; j < dst.grid.shape[axis]; ++j)
        {
            // Set floating axis
            dst_tile_index[axis] = j;
            // Get linear offset from index
            Index dst_tile_offset = dst.grid.index_to_linear(dst_tile_index);
            // Get destination tile traits
            auto dst_tile_traits = dst.get_tile_traits(dst_tile_offset);
            // Get destination tile handle
            auto dst_tile_handle = dst.get_tile_handle(dst_tile_offset);
            // MPI rank of the destination tile
            int dst_tile_rank = starpu_mpi_data_get_rank(dst_tile_handle);
            // Transfer data
            if(mpi_rank == src_tile_rank and mpi_rank != dst_tile_rank)
            {
                ret = starpu_mpi_isend_detached(src_tile_handle, dst_tile_rank,
                        tile_tag, MPI_COMM_WORLD, nullptr, nullptr);
                if(ret != 0)
                {
                    throw std::runtime_error("Error in starpu_mpi_isend_"
                            "detached");
                }
            }
            if(mpi_rank == dst_tile_rank and mpi_rank != src_tile_rank)
            {
                ret = starpu_mpi_irecv_detached(src_tile_handle, src_tile_rank,
                        tile_tag, MPI_COMM_WORLD, nullptr, nullptr);
                if(ret != 0)
                {
                    throw std::runtime_error("Error in starpu_mpi_irecv_"
                            "detached");
                }
            }
            // Execute on destination node
            if(mpi_rank == dst_tile_rank)
            {
                // Reshape inputs: src_tile -> (m,n), dst_tile -> (m,k,n)
                Index m, n, k;
                if(axis == 0)
                {
                    m = 1;
                    n = src_tile_traits.nelems;
                    k = dst_tile_traits.shape[0];
                }
                else if(axis == dst.ndim-1)
                {
                    m = src_tile_traits.nelems;
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
                starpu::bias::submit<T>(m, n, k, src_tile_handle,
                        dst_tile_handle);
            }
        }
    }
}

//! Tensor-wise bias operation
template<typename T>
void bias(const Tensor<T> &src, const Tensor<T> &dst, Index axis)
{
    bias_async<T>(src, dst, axis);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void bias(const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst, Index axis);

template
void bias(const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst, Index axis);

} // namespace tensor
} // namespace nntile

