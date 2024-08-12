/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/add_slice3.cc
 * Tensor wrappers for addition of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/add_slice3.hh"
#include "nntile/starpu/add_slice3.hh"

namespace nntile::tensor
{

template<typename T>
void add_slice3_async(Scalar alpha, const Tensor<T> &src1, Scalar beta,
        const Tensor<T> &src2, const Tensor<T> &dst, Index axis)
//! Tensor<T> addition of a tensor and a broadcasted slice
/*! Reshapes input tensor and slice into 3-dimensional and 2-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha*src1[i,j] + beta*src2[i,l,j]
 *
 * @param[in] alpha: Scalar factor for src1
 * @param[in] src1: Input slice, that is reshaped into 2D array
 * @param[in] beta: Scaling factor for src2
 * @param[in] src2: Input tensor
 * @param[out] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    // Check dimensions
    if(dst.ndim != src1.ndim+1)
    {
        throw std::runtime_error("dst.ndim != src1.ndim+1");
    }
    if(dst.ndim != src2.ndim)
    {
        throw std::runtime_error("dst.ndim != src2.ndim");
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
        if(dst.shape[i] != src1.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src1.shape[i]");
        }
        if(dst.basetile_shape[i] != src1.basetile_shape[i])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "src1.basetile_shape[i]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src1.shape[i-1])
        {
            throw std::runtime_error("dst.shape[i] != src1.shape[i-1]");
        }
        if(dst.basetile_shape[i] != src1.basetile_shape[i-1])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "src1.basetile_shape[i-1]");
        }
    }
    if(dst.shape != src2.shape)
    {
        throw std::runtime_error("dst.shape != src2.shape");
    }
    if(dst.basetile_shape != src2.basetile_shape)
    {
        throw std::runtime_error("dst.basetile_shape != src2.basetile_shape");
    }
    // Apply per-tile add_slice3 asynchronously as needed
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    for(Index i = 0; i < src1.grid.nelems; ++i)
    {
        // Index of current source tile
        auto src1_tile_index = src1.grid.linear_to_index(i);
        // Source tile traits
        auto src1_tile_traits = src1.get_tile_traits(i);
        // Source tile handle
        auto src1_tile_handle = src1.get_tile_handle(i);
        // Set fixed indices of current destination tile
        std::vector<Index> dst_tile_index(dst.ndim);
        for(Index j = 0; j < axis; ++j)
        {
            dst_tile_index[j] = src1_tile_index[j];
        }
        for(Index j = axis+1; j < dst.ndim; ++j)
        {
            dst_tile_index[j] = src1_tile_index[j-1];
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
            // Get src2 tile handle
            auto src2_tile_handle = src2.get_tile_handle(dst_tile_offset);
            // MPI rank of the destination tile
            int dst_tile_rank = dst_tile_handle.mpi_get_rank();
            // Transfer data
            src1_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
            src2_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
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
                starpu::add_slice3::submit<T>(m, n, k, alpha, src1_tile_handle,
                        beta, src2_tile_handle, dst_tile_handle);
            }
            // Flush cache for the output tile on every node
            dst_tile_handle.mpi_flush();
        }
    }
}

template<typename T>
void add_slice3(Scalar alpha, const Tensor<T> &src1, Scalar beta, const Tensor<T> &src2,
        const Tensor<T> &dst, Index axis)
//! Tensor<T> addition of a tensor and a broadcasted slice
/*! Blocking version of add_slice3_async<T>.
 * Reshapes input tensor and slice into 3-dimensional and 2-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha*src1[i,j] + beta*src2[i,l,j]
 *
 * @param[in] alpha: Scalar factor for src1
 * @param[in] src1: Input slice, that is reshaped into 2D array
 * @param[in] beta: Scaling factor for src2
 * @param[in] src2: Input tensor
 * @param[out] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    add_slice3_async<T>(alpha, src1, beta, src2, dst, axis);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void add_slice3_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src1,
        Scalar beta, const Tensor<fp32_t> &src2, const Tensor<fp32_t> &dst,
        Index axis);

template
void add_slice3_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src1,
        Scalar beta, const Tensor<fp32_fast_tf32_t> &src2, const Tensor<fp32_fast_tf32_t> &dst,
        Index axis);

template
void add_slice3_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src,
        Scalar beta, const Tensor<fp64_t> &src2, const Tensor<fp64_t> &dst,
        Index axis);

template
void add_slice3_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src1, Scalar beta,
        const Tensor<bf16_t> &src2, const Tensor<bf16_t> &dst, Index axis);

// Explicit instantiation of template
template
void add_slice3<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src1, Scalar beta,
        const Tensor<fp32_t> &src2, const Tensor<fp32_t> &dst, Index axis);

template
void add_slice3<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src1, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &src2, const Tensor<fp32_fast_tf32_t> &dst, Index axis);

template
void add_slice3<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src1, Scalar beta,
        const Tensor<fp64_t> &src2, const Tensor<fp64_t> &dst, Index axis);

template
void add_slice3<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src1, Scalar beta,
        const Tensor<bf16_t> &src2, const Tensor<bf16_t> &dst, Index axis);

} // namespace nntile::tensor
