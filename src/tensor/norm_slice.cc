/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/norm_slice.cc
 * Euclidean norms of fibers into a slice of a Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/norm_slice.hh"
#include "nntile/starpu/norm_slice.hh"

namespace nntile::tensor
{

template<typename T>
void norm_slice_async(Scalar alpha, const Tensor<T> &src, Scalar beta,
        const Tensor<T> &dst, Index axis, int redux)
{
    // Check dimensions
    if(src.ndim-1 != dst.ndim)
    {
        throw std::runtime_error("src.ndim-1 != dst.ndim");
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
    // Check shapes of src and dst
    for(Index i = 0; i < axis; i++)
    {
        if(src.shape[i] != dst.shape[i])
        {
            throw std::runtime_error("src.shape[i] != dst.shape[i]");
        }
        if(src.basetile_shape[i] != dst.basetile_shape[i])
        {
            throw std::runtime_error("src.basetile_shape[i] != "
                    "dst.basetile_shape[i]");
        }
    }
    for(Index i = axis+1; i < ndim; i++)
    {
        if(src.shape[i] != dst.shape[i-1])
        {
            throw std::runtime_error("src.shape[i] != dst.shape[i-1]");
        }
        if(src.basetile_shape[i] != dst.basetile_shape[i-1])
        {
            throw std::runtime_error("src.basetile_shape[i] != "
                    "dst.basetile_shape[i-1]");
        }
    }
    // Do actual calculations
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    constexpr Scalar one = 1.0;
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        auto dst_tile_handle = dst.get_tile_handle(i);
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Obtain indices of applicable source tiles
        auto dst_tile_index = dst.grid.linear_to_index(i);
        std::vector<Index> src_tile_index(src.ndim);
        for(Index j = 0, k = 0; j < src.ndim; ++j)
        {
            if(j == axis)
            {
                src_tile_index[axis] = 0;
                continue;
            }
            src_tile_index[j] = dst_tile_index[k];
            ++k;
        }
        auto dst_tile_traits = dst.get_tile_traits(i);
        // Launch kernel for each appropriate source tile
        for(Index j = 0; j < src.grid.shape[axis]; ++j)
        {
            src_tile_index[axis] = j;
            Index src_tile_offset = src.grid.index_to_linear(src_tile_index);
            auto src_tile_handle = src.get_tile_handle(src_tile_offset);
            int src_tile_rank = src_tile_handle.mpi_get_rank();
            // Transfer data
            src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
            // Execute on destination node
            if(mpi_rank == dst_tile_rank)
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
                    starpu::norm_slice::submit<T>(m, n, k, alpha,
                            src_tile_handle, beta, dst_tile_handle, redux);
                }
                // Insert all consequent tasks
                else
                {
                    starpu::norm_slice::submit<T>(m, n, k, alpha,
                            src_tile_handle, one, dst_tile_handle, redux);
                }
            }
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

template<typename T>
void norm_slice(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst,
        Index axis, int redux)
{
    norm_slice_async<T>(alpha, src, beta, dst, axis, redux);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void norm_slice_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src,
        Scalar beta, const Tensor<fp32_t> &dst, Index axis, int redux);

template
void norm_slice_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src,
        Scalar beta, const Tensor<fp32_fast_tf32_t> &dst, Index axis, int redux);

template
void norm_slice_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src,
        Scalar beta, const Tensor<fp64_t> &dst, Index axis, int redux);

template
void norm_slice_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src, Scalar beta,
        const Tensor<bf16_t> &dst, Index axis, int redux);

// Explicit instantiation
template
void norm_slice<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src, Scalar beta,
        const Tensor<fp32_t> &dst, Index axis, int redux);

template
void norm_slice<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &dst, Index axis, int redux);

template
void norm_slice<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src, Scalar beta,
        const Tensor<fp64_t> &dst, Index axis, int redux);

template
void norm_slice<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src, Scalar beta,
        const Tensor<bf16_t> &dst, Index axis, int redux);

} // namespace nntile::tensor
