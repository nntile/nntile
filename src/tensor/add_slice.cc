/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/add_slice.cc
 * Bias operation over fibers from a slice of a Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-26
 * */

#include "nntile/tensor/add_slice.hh"
#include "nntile/starpu/add_slice.hh"

namespace nntile
{
namespace tensor
{

//! Tensor-wise add_slice operation
template<typename T>
void add_slice_async(T alpha, const Tensor<T> &src, T beta,
        const Tensor<T> &dst, Index axis)
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
    // Apply per-tile add_slice asynchronously as needed
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
            // Get destination tile handle
            auto dst_tile_handle = dst.get_tile_handle(dst_tile_offset);
            // MPI rank of the destination tile
            int dst_tile_rank = dst_tile_handle.mpi_get_rank();
            // Transfer data
            src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
            // Execute on destination node
            if(mpi_rank == dst_tile_rank)
            {
                // Get destination tile traits
                auto dst_tile_traits = dst.get_tile_traits(dst_tile_offset);
                // Reshape inputs: src_tile -> (m,n), dst_tile -> (m,k,n)
                Index m, n, k;
                m = dst_tile_traits.stride[axis];
                n = dst_tile_traits.matrix_shape[axis+1][1];
                k = dst_tile_traits.shape[axis];
                // Insert corresponding task
                starpu::add_slice::submit<T>(m, n, k, alpha, src_tile_handle,
                        beta, dst_tile_handle);
            }
            // Flush cache for the output tile on every node
            dst_tile_handle.mpi_flush();
        }
    }
}

//! Tensor-wise add_slice operation
template<typename T>
void add_slice(T alpha, const Tensor<T> &src, T beta, const Tensor<T> &dst,
        Index axis)
{
    add_slice_async<T>(alpha, src, beta, dst, axis);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void add_slice_async<fp32_t>(fp32_t alpha, const Tensor<fp32_t> &src,
        fp32_t beta, const Tensor<fp32_t> &dst, Index axis);

template
void add_slice_async<fp64_t>(fp64_t alpha, const Tensor<fp64_t> &src,
        fp64_t beta, const Tensor<fp64_t> &dst, Index axis);

// Explicit instantiation of template
template
void add_slice<fp32_t>(fp32_t alpha, const Tensor<fp32_t> &src, fp32_t beta,
        const Tensor<fp32_t> &dst, Index axis);

template
void add_slice<fp64_t>(fp64_t alpha, const Tensor<fp64_t> &src, fp64_t beta,
        const Tensor<fp64_t> &dst, Index axis);

} // namespace tensor
} // namespace nntile

