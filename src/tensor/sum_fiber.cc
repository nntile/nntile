/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/sum_fiber.cc
 * Sum over fibers into a slice of a Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-26
 * */

#include "nntile/tensor/sum_fiber.hh"
#include "nntile/starpu/sum_fiber.hh"

namespace nntile
{
namespace tensor
{

//! Tensor-wise sum_fiber
template<typename T>
void sum_fiber_async(T alpha, const Tensor<T> &src, T beta,
        const Tensor<T> &dst, Index axis)
{
    // Check dimensions
    if(dst.ndim != 1)
    {
        throw std::runtime_error("dst.ndim != 1");
    }
    // Treat special case of src.ndim=0
    if(src.ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= src.ndim)
    {
        throw std::runtime_error("axis >= src.ndim");
    }
    // Check shapes
    if(dst.shape[0] != src.shape[axis])
    {
        throw std::runtime_error("dst.shape[0] != src.shape[axis]");
    }
    if(dst.basetile_shape[0] != src.basetile_shape[axis])
    {
        throw std::runtime_error("dst.basetile_shape[0] != "
                "src.basetile_shape[axis]");
    }
    // Do actual calculations
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    Index ndim = src.ndim;
    constexpr T one = 1.0;
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        auto src_tile_handle = src.get_tile_handle(i);
        auto src_tile_traits = src.get_tile_traits(i);
        int src_tile_rank = src_tile_handle.mpi_get_rank();
        auto src_tile_index = src.grid.linear_to_index(i);
        // Get corresponding dst tile
        Index j = src_tile_index[axis];
        auto dst_tile_handle = dst.get_tile_handle(j);
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Transfer data
        src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        // Execute on destination node
        if(mpi_rank == dst_tile_rank)
        {
            // Get sizes
            Index m, n, k;
            m = src_tile_traits.stride[axis];
            n = src_tile_traits.matrix_shape[axis+1][1];
            k = src_tile_traits.shape[axis];
            // Check if it is the first task for the output tile
            bool init_first = true;
            for(Index j = 0; j < src.ndim; ++j)
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
                starpu::sum_fiber::submit<T>(m, n, k, alpha, src_tile_handle,
                        beta, dst_tile_handle);
            }
            else
            {
                starpu::sum_fiber::submit<T>(m, n, k, alpha, src_tile_handle,
                        one, dst_tile_handle);
            }
        }
    }
    // Flush cache for the output tiles on every node
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        dst.get_tile_handle(i).mpi_flush();
    }
}

//! Tensor-wise sum_fiber
template<typename T>
void sum_fiber(T alpha, const Tensor<T> &src, T beta, const Tensor<T> &dst,
        Index axis)
{
    sum_fiber_async<T>(alpha, src, beta, dst, axis);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void sum_fiber_async<fp32_t>(fp32_t alpha, const Tensor<fp32_t> &src,
        fp32_t beta, const Tensor<fp32_t> &dst, Index axis);

template
void sum_fiber_async<fp64_t>(fp64_t alpha, const Tensor<fp64_t> &src,
        fp64_t beta, const Tensor<fp64_t> &dst, Index axis);

// Explicit instantiation
template
void sum_fiber<fp32_t>(fp32_t alpha, const Tensor<fp32_t> &src, fp32_t beta,
        const Tensor<fp32_t> &dst, Index axis);

template
void sum_fiber<fp64_t>(fp64_t alpha, const Tensor<fp64_t> &src, fp64_t beta,
        const Tensor<fp64_t> &dst, Index axis);

} // namespace tensor
} // namespace nntile

