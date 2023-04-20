/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/biasprod_outer.cc
 * Bias-like product along outer axes operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-20
 * */

#include "nntile/tensor/biasprod_outer.hh"
#include "nntile/starpu/biasprod_outer.hh"

namespace nntile
{
namespace tensor
{

//! Tensor-wise biasprod_outer operation
template<typename T>
void biasprod_outer_async(const Tensor<T> &src, const Tensor<T> &dst,
        Index axis)
{
    // Check dimensions
    if(src.ndim != 1)
    {
        throw std::runtime_error("src.ndim != 1");
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
    if(src.shape[0] != dst.shape[axis])
    {
        throw std::runtime_error("src.shape[0] != dst.shape[axis]");
    }
    if(src.basetile_shape[0] != dst.basetile_shape[axis])
    {
        throw std::runtime_error("src.basetile_shape[0] != "
                "dst.basetile_shape[axis]");
    }
    // Apply per-tile biasprod_outer asynchronously as needed
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        auto dst_tile_index = dst.grid.linear_to_index(i);
        auto dst_tile_traits = dst.get_tile_traits(i);
        auto dst_tile_handle = dst.get_tile_handle(i);
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Get corresponding src tile
        Index j = dst_tile_index[axis];
        auto src_tile_handle = src.get_tile_handle(j);
        int src_tile_rank = src_tile_handle.mpi_get_rank();
        // Transfer data
        src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        // Execute on destination node
        if(mpi_rank == dst_tile_rank)
        {
            // Reshape inputs: src_tile -> (m,n), dst_tile -> (m,k,n)
            Index m, n, k;
            m = dst_tile_traits.stride[axis];
            n = dst_tile_traits.matrix_shape[axis+1][1];
            k = dst_tile_traits.shape[axis];
            // Insert corresponding task
            starpu::biasprod_outer::submit<T>(m, n, k, src_tile_handle,
                    dst_tile_handle);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

//! Tensor-wise biasprod_outer operation
template<typename T>
void biasprod_outer(const Tensor<T> &src, const Tensor<T> &dst, Index axis)
{
    biasprod_outer_async<T>(src, dst, axis);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void biasprod_outer_async<fp32_t>(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst, Index axis);

template
void biasprod_outer_async<fp64_t>(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst, Index axis);

// Explicit instantiation of template
template
void biasprod_outer<fp32_t>(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst, Index axis);

template
void biasprod_outer<fp64_t>(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst, Index axis);

} // namespace tensor
} // namespace nntile

