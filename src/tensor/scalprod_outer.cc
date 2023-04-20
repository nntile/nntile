/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/scalprod_outer.cc
 * Scalar products along outer axes of two Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-20
 * */

#include "nntile/tensor/scalprod_outer.hh"
#include "nntile/starpu/scalprod_outer.hh"

namespace nntile
{
namespace tensor
{

//! Compute scalar products of slices along outer axes
template<typename T>
void scalprod_outer_async(T alpha, const Tensor<T> &src1,
        const Tensor<T> &src2, T beta, const Tensor<T> &dst, Index axis)
{
    // Check shapes of src1 and src2
    if(src1.shape != src2.shape)
    {
        throw std::runtime_error("src1.shape != src2.shape");
    }
    // Check dimensions
    if(dst.ndim != 1)
    {
        throw std::runtime_error("dst.ndim != 1");
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
    // Check shapes of src1 and dst
    if(src1.shape[axis] != dst.shape[0])
    {
        throw std::runtime_error("src1.shape[axis] != dst.shape[0]");
    }
    if(src1.basetile_shape[axis] != dst.basetile_shape[0])
    {
        throw std::runtime_error("src1.basetile_shape[axis] != "
                "dst.basetile_shape[0]");
    }
    // Do actual calculations
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    Index ndim = src1.ndim;
    constexpr T one = 1.0;
    for(Index i = 0; i < src1.grid.nelems; ++i)
    {
        // Get source tiles
        auto src1_tile_handle = src1.get_tile_handle(i);
        auto src2_tile_handle = src2.get_tile_handle(i);
        auto src_tile_traits = src1.get_tile_traits(i);
        int src1_tile_rank = src1_tile_handle.mpi_get_rank();
        int src2_tile_rank = src2_tile_handle.mpi_get_rank();
        auto src_tile_index = src1.grid.linear_to_index(i);
        // Get destination tile
        Index j = src_tile_index[axis];
        auto dst_tile_handle = dst.get_tile_handle(j);
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Transfer data
        src1_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        src2_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        // Execute on destination node
        auto dst_tile_traits = dst.get_tile_traits(i);
        if(mpi_rank == dst_tile_rank)
        {
            // Get sizes
            Index m, n, k;
            m = src_tile_traits.stride[axis];
            n = src_tile_traits.matrix_shape[axis+1][1];
            k = src_tile_traits.shape[axis];
            // Check if it is the first task for the output tile
            bool init_first = true;
            for(Index j = 0; j < src1.ndim; ++j)
            {
                if(j != axis and src_tile_index[j] != 0)
                {
                    init_first = false;
                    break;
                }
            }
            // Insert task
            if(init_first)
            {
                starpu::scalprod_outer::submit<T>(m, n, k, alpha,
                        src1_tile_handle, src2_tile_handle, beta,
                        dst_tile_handle);
            }
            else
            {
                starpu::scalprod_outer::submit<T>(m, n, k, alpha,
                        src1_tile_handle, src2_tile_handle, one,
                        dst_tile_handle);
            }
        }
    }
    // Flush cache for the output tiles on every node
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        dst.get_tile_handle(i).mpi_flush();
    }
}


template<typename T>
void scalprod_outer(T alpha, const Tensor<T> &src1, const Tensor<T> &src2,
        T beta, const Tensor<T> &dst, Index axis)
{
    scalprod_outer_async<T>(alpha, src1, src2, beta, dst, axis);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void scalprod_outer_async<fp32_t>(fp32_t alpha, const Tensor<fp32_t> &src1,
        const Tensor<fp32_t> &src2, fp32_t beta, const Tensor<fp32_t> &dst,
        Index axis);

template
void scalprod_outer_async<fp64_t>(fp64_t alpha, const Tensor<fp64_t> &src1,
        const Tensor<fp64_t> &src2, fp64_t beta, const Tensor<fp64_t> &dst,
        Index axis);

// Explicit instantiation
template
void scalprod_outer<fp32_t>(fp32_t alpha, const Tensor<fp32_t> &src1,
        const Tensor<fp32_t> &src2, fp32_t beta, const Tensor<fp32_t> &dst,
        Index axis);

template
void scalprod_outer<fp64_t>(fp64_t alpha, const Tensor<fp64_t> &src1,
        const Tensor<fp64_t> &src2, fp64_t beta, const Tensor<fp64_t> &dst,
        Index axis);

} // namespace tensor
} // namespace nntile

