/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/norm.cc
 * Norm of Tensor<T> along axis
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-18
 * */

#include "nntile/tensor/norm.hh"
#include "nntile/starpu/norm.hh"

namespace nntile
{
namespace tensor
{

//! Compute norms of slices along given axis
template<typename T>
void norm_async(T alpha, const Tensor<T> &src, T beta,
        const Tensor<T> &norm_dst, Index axis)
{
    // Check dimensions
    if(src.ndim-1 != norm_dst.ndim)
    {
        throw std::runtime_error("src.ndim-1 != norm_dst.ndim");
    }
    // Treat special case of src.ndim=0
    if(src.ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    Index ndim = src.ndim;
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= src.ndim)
    {
        throw std::runtime_error("axis >= src.ndim");
    }
    // Check shapes of src and norm_dst
    for(Index i = 0; i < axis; i++)
    {
        if(src.shape[i] != norm_dst.shape[i])
        {
            throw std::runtime_error("src.shape[i] != norm_dst.shape[i]");
        }
        if(src.basetile_shape[i] != norm_dst.basetile_shape[i])
        {
            throw std::runtime_error("src.basetile_shape[i] != "
                    "norm_dst.basetile_shape[i]");
        }
    }
    for(Index i = axis+1; i < ndim; i++)
    {
        if(src.shape[i] != norm_dst.shape[i-1])
        {
            throw std::runtime_error("src.shape[i] != norm_dst.shape[i-1]");
        }
        if(src.basetile_shape[i] != norm_dst.basetile_shape[i-1])
        {
            throw std::runtime_error("src.basetile_shape[i] != "
                    "norm_dst.basetile_shape[i-1]");
        }
    }
    // Do actual calculations
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    constexpr T one = 1.0;
    for(Index i = 0; i < norm_dst.grid.nelems; ++i)
    {
        auto norm_dst_tile_handle = norm_dst.get_tile_handle(i);
        int norm_dst_tile_rank = norm_dst_tile_handle.mpi_get_rank();
        // Obtain indices of applicable source tiles
        auto norm_dst_tile_index = norm_dst.grid.linear_to_index(i);
        std::vector<Index> src_tile_index(src.ndim);
        for(Index j = 0, k = 0; j < src.ndim; ++j)
        {
            if(j == axis)
            {
                src_tile_index[axis] = 0;
                continue;
            }
            src_tile_index[j] = norm_dst_tile_index[k];
            ++k;
        }
        auto norm_dst_tile_traits = norm_dst.get_tile_traits(i);
        // Launch kernel for each appropriate source tile
        for(Index j = 0; j < src.grid.shape[axis]; ++j)
        {
            src_tile_index[axis] = j;
            Index src_tile_offset = src.grid.index_to_linear(src_tile_index);
            auto src_tile_handle = src.get_tile_handle(src_tile_offset);
            int src_tile_rank = src_tile_handle.mpi_get_rank();
            // Transfer data
            src_tile_handle.mpi_transfer(norm_dst_tile_rank, mpi_rank);
            // Execute on destination node
            if(mpi_rank == norm_dst_tile_rank)
            {
                // Get sizes
                auto src_tile_traits = src.get_tile_traits(src_tile_offset);
                Index m, n, k;
                m = src_tile_traits.stride[axis];
                n = src_tile_traits.matrix_shape[axis+1][1];
                k = src_tile_traits.shape[axis];
                // Insert initial task
                if(j == 0)
                {
                    starpu::norm::submit<T>(m, n, k, alpha, src_tile_handle,
                            beta, norm_dst_tile_handle);
                }
                // Insert all consequent tasks
                else
                {
                    starpu::norm::submit<T>(m, n, k, alpha, src_tile_handle,
                            one, norm_dst_tile_handle);
                }
            }
        }
        // Flush cache for the output tile on every node
        norm_dst_tile_handle.mpi_flush();
    }
}


template<typename T>
void norm(T alpha, const Tensor<T> &src, T beta, const Tensor<T> &norm_dst,
        Index axis)
{
    norm_async<T>(alpha, src, beta, norm_dst, axis);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void norm_async<fp32_t>(fp32_t alpha, const Tensor<fp32_t> &src, fp32_t beta,
        const Tensor<fp32_t> &norm_dst, Index axis);

template
void norm_async<fp64_t>(fp64_t alpha, const Tensor<fp64_t> &src, fp64_t beta,
        const Tensor<fp64_t> &norm_dst, Index axis);

// Explicit instantiation
template
void norm<fp32_t>(fp32_t alpha, const Tensor<fp32_t> &src, fp32_t beta,
        const Tensor<fp32_t> &norm_dst, Index axis);

template
void norm<fp64_t>(fp64_t alpha, const Tensor<fp64_t> &src, fp64_t beta,
        const Tensor<fp64_t> &norm_dst, Index axis);

} // namespace tensor
} // namespace nntile

