/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/norm_fiber.cc
 * Euclidean norms over slices into a fiber of a product of a Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/norm_fiber.hh"
#include "nntile/starpu/norm_fiber.hh"

namespace nntile::tensor
{

//! Tensor-wise norm_fiber
template<typename T>
void norm_fiber_async(Scalar alpha, const Tensor<T> &src, Scalar beta,
        const Tensor<T> &dst, Index axis, Index batch_ndim, int redux)
{
    // Check dimensions
    if(dst.ndim != batch_ndim+1)
    {
        throw std::runtime_error("dst.ndim != batch_ndim+1");
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
    if(axis >= src.ndim-batch_ndim)
    {
        throw std::runtime_error("axis >= src.ndim-batch_ndim");
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
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(dst.shape[i+1] != src.shape[src.ndim-batch_ndim+i])
        {
            throw std::runtime_error("dst.shape[i+1] != "
                    "src.shape[src.ndim-batch_ndim+i]");
        }
        if(dst.basetile_shape[i+1] != src.basetile_shape[src.ndim-batch_ndim+i])
        {
            throw std::runtime_error("dst.basetile_shape[i+1] != "
                    "src.basetile_shape[src.ndim-batch_ndim+i]");
        }
    }
    // Do actual calculations
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    Index ndim = src.ndim;
    constexpr Scalar one = 1.0;
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        auto src_tile_handle = src.get_tile_handle(i);
        auto src_tile_traits = src.get_tile_traits(i);
        int src_tile_rank = src_tile_handle.mpi_get_rank();
        auto src_tile_index = src.grid.linear_to_index(i);
        // Get corresponding dst tile
        std::vector<Index> dst_tile_index(dst.ndim);
        dst_tile_index[0] = src_tile_index[axis];
        for(Index j = 0; j < batch_ndim; ++j)
        {
            dst_tile_index[j+1] = src_tile_index[src.ndim-batch_ndim+j];
        }
        auto dst_tile_handle = dst.get_tile_handle(dst_tile_index);
        auto dst_tile_traits = dst.get_tile_traits(dst_tile_index);
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Transfer data
        src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        // Execute on destination node
        if(mpi_rank == dst_tile_rank)
        {
            // Get sizes
            Index m, n, k, batch;
            batch = dst_tile_traits.matrix_shape[1][1];
            m = src_tile_traits.stride[axis];
            n = src_tile_traits.matrix_shape[axis+1][1] / batch;
            k = src_tile_traits.shape[axis];
            // Check if it is the first task for the output tile
            bool init_first = true;
            for(Index j = 0; j < src.ndim-batch_ndim; ++j)
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
                starpu::norm_fiber::submit<T>(m, n, k, batch, alpha,
                        src_tile_handle, beta, dst_tile_handle);
            }
            else
            {
                starpu::norm_fiber::submit<T>(m, n, k, batch, alpha,
                        src_tile_handle, one, dst_tile_handle, redux);
            }
        }
    }
    // Flush cache for the output tiles on every node
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        dst.get_tile_handle(i).mpi_flush();
    }
}

//! Tensor-wise norm_fiber
template<typename T>
void norm_fiber(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst,
        Index axis, Index batch_ndim, int redux)
{
    norm_fiber_async<T>(alpha, src, beta, dst, axis, batch_ndim, redux);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void norm_fiber_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src,
        Scalar beta, const Tensor<fp32_t> &dst, Index axis, Index batch_ndim,
        int redux);

template
void norm_fiber_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src,
        Scalar beta, const Tensor<fp32_fast_tf32_t> &dst, Index axis, Index batch_ndim,
        int redux);

template
void norm_fiber_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src,
        Scalar beta, const Tensor<fp64_t> &dst, Index axis, Index batch_ndim,
        int redux);

template
void norm_fiber_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src, Scalar beta,
        const Tensor<bf16_t> &dst, Index axis, Index batch_ndim,
        int redux);

// Explicit instantiation
template
void norm_fiber<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src, Scalar beta,
        const Tensor<fp32_t> &dst, Index axis, Index batch_ndim,
        int redux);

template
void norm_fiber<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &dst, Index axis, Index batch_ndim,
        int redux);

template
void norm_fiber<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src, Scalar beta,
        const Tensor<fp64_t> &dst, Index axis, Index batch_ndim,
        int redux);

template
void norm_fiber<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src, Scalar beta,
        const Tensor<bf16_t> &dst, Index axis, Index batch_ndim,
        int redux);

} // namespace nntile::tensor
