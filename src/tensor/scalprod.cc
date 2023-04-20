/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/scalprod.cc
 * Scalar products of two Tensor<T> along axis
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-20
 * */

#include "nntile/tensor/scalprod.hh"
#include "nntile/starpu/scalprod.hh"
#include "nntile/starpu/clear.hh"

namespace nntile
{
namespace tensor
{

//! Compute scalar products of slices along given axis
template<typename T>
void scalprod_async(T alpha, const Tensor<T> &src1, const Tensor<T> &src2,
        T beta, const Tensor<T> &dst, Index axis)
{
    // Check shapes of src1 and src2
    if(src1.shape != src2.shape)
    {
        throw std::runtime_error("src1.shape != src2.shape");
    }
    // Check dimensions
    if(src1.ndim != dst.ndim+1)
    {
        throw std::runtime_error("src1.ndim != dst.ndim+1");
    }
    // Treat special case of src.ndim=0
    if(src1.ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= src1.ndim)
    {
        throw std::runtime_error("axis >= src1.ndim");
    }
    // Check shapes of src and dst
    for(Index i = 0; i < axis; ++i)
    {
        if(src1.shape[i] != dst.shape[i])
        {
            throw std::runtime_error("src1.shape[i] != dst.shape[i]");
        }
        if(src1.basetile_shape[i] != dst.basetile_shape[i])
        {
            throw std::runtime_error("src1.basetile_shape[i] != "
                    "dst.basetile_shape[i]");
        }
    }
    for(Index i = axis+1; i < src1.ndim; ++i)
    {
        if(src1.shape[i] != dst.shape[i-1])
        {
            throw std::runtime_error("src1.shape[i] != dst.shape[i-1]");
        }
        if(src1.basetile_shape[i] != dst.basetile_shape[i-1])
        {
            throw std::runtime_error("src1.basetile_shape[i] != "
                    "dst.basetile_shape[i-1]");
        }
    }
    // Do actual calculations
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    Index ndim = src1.ndim;
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        // Get destination tile
        auto dst_tile_handle = dst.get_tile_handle(i);
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Obtain indices of applicable source tiles
        auto dst_tile_index = dst.grid.linear_to_index(i);
        std::vector<Index> src_tile_index(src1.ndim);
        for(Index j = 0; j < axis; ++j)
        {
            src_tile_index[j] = dst_tile_index[j];
        }
        src_tile_index[axis] = 0;
        for(Index j = axis+1; j < src1.ndim; ++j)
        {
            src_tile_index[j] = dst_tile_index[j-1];
        }
        // Initialize with provided beta only single time
        Index src_tile_offset = src1.grid.index_to_linear(src_tile_index);
        auto src1_tile_handle = src1.get_tile_handle(src_tile_offset);
        int src1_tile_rank = src1_tile_handle.mpi_get_rank();
        auto src2_tile_handle = src2.get_tile_handle(src_tile_offset);
        int src2_tile_rank = src2_tile_handle.mpi_get_rank();
        // Transfer data
        src1_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        src2_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        // Execute on destination node
        auto dst_tile_traits = dst.get_tile_traits(i);
        if(mpi_rank == dst_tile_rank)
        {
            // Get sizes
            auto src_tile_traits = src1.get_tile_traits(src_tile_offset);
            Index m, n, k;
            m = src_tile_traits.stride[axis];
            n = src_tile_traits.matrix_shape[axis+1][1];
            k = src_tile_traits.shape[axis];
            // Insert task
            starpu::scalprod::submit<T>(m, n, k, alpha, src1_tile_handle,
                    src2_tile_handle, beta, dst_tile_handle);
        }
        // Launch kernel for all other appropriate source tiles with beta=1
        for(Index j = 1; j < src1.grid.shape[axis]; ++j)
        {
            src_tile_index[axis] = j;
            Index src_tile_offset = src1.grid.index_to_linear(src_tile_index);
            auto src1_tile_handle = src1.get_tile_handle(src_tile_offset);
            int src1_tile_rank = src1_tile_handle.mpi_get_rank();
            auto src2_tile_handle = src2.get_tile_handle(src_tile_offset);
            int src2_tile_rank = src2_tile_handle.mpi_get_rank();
            // Transfer data
            src1_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
            src2_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
            // Execute on destination node
            if(mpi_rank == dst_tile_rank)
            {
                // Get sizes
                auto src_tile_traits = src1.get_tile_traits(src_tile_offset);
                Index m, n, k;
                if(axis == 0)
                {
                    m = 1;
                    n = dst_tile_traits.nelems;
                    k = src_tile_traits.shape[0];
                }
                else if(axis == ndim-1)
                {
                    m = dst_tile_traits.nelems;
                    n = 1;
                    k = src_tile_traits.shape[axis];
                }
                else
                {
                    m = src_tile_traits.stride[axis];
                    n = src_tile_traits.matrix_shape[axis+1][1];
                    k = src_tile_traits.shape[axis];
                }
                // Insert task
                starpu::scalprod::submit<T>(m, n, k, alpha, src1_tile_handle,
                        src2_tile_handle, 1.0, dst_tile_handle);
            }
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}


template<typename T>
void scalprod(T alpha, const Tensor<T> &src1, const Tensor<T> &src2, T beta,
        const Tensor<T> &dst, Index axis)
{
    scalprod_async<T>(alpha, src1, src2, beta, dst, axis);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void scalprod_async<fp32_t>(fp32_t alpha, const Tensor<fp32_t> &src1,
        const Tensor<fp32_t> &src2, fp32_t beta, const Tensor<fp32_t> &dst,
        Index axis);

template
void scalprod_async<fp64_t>(fp64_t alpha, const Tensor<fp64_t> &src1,
        const Tensor<fp64_t> &src2, fp64_t beta, const Tensor<fp64_t> &dst,
        Index axis);

// Explicit instantiation
template
void scalprod<fp32_t>(fp32_t alpha, const Tensor<fp32_t> &src1,
        const Tensor<fp32_t> &src2, fp32_t beta, const Tensor<fp32_t> &dst,
        Index axis);

template
void scalprod<fp64_t>(fp64_t alpha, const Tensor<fp64_t> &src1,
        const Tensor<fp64_t> &src2, fp64_t beta, const Tensor<fp64_t> &dst,
        Index axis);

} // namespace tensor
} // namespace nntile

