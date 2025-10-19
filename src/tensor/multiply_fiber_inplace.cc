/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/prod_fiber.cc
 * Tensor wrappers for per-element product of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/multiply_fiber_inplace.hh"
#include "nntile/starpu/multiply_fiber_inplace.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

template<typename T>
void multiply_fiber_inplace_async(Scalar alpha, const Tensor<T> &src, const Tensor<T> &dst,
        Index axis)
//! Tensor<T> per-element multiplication of a tensor and a broadcasted fiber
/*! Reshapes input tensor and fiber into 3-dimensional and 1-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha * dst[i,l,j] * src[l]
 *
 * @param[in] alpha: Scalar factor
 * @param[in] src: Input fiber, that is reshaped into 1D array
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
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
    // Apply per-tile multiply_fiber_inplace asynchronously as needed
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
            starpu::multiply_fiber_inplace.submit<std::tuple<T>>(m, n, k, alpha, src_tile_handle,
                    dst_tile_handle);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

template<typename T>
void multiply_fiber_inplace(Scalar alpha, const Tensor<T> &src, const Tensor<T> &dst,
        Index axis)
//! Tensor<T> per-element multiplication of a tensor and a broadcasted fiber
/*! Blocking version of multiply_fiber_inplace_async<T>.
 * Reshapes input tensor and fiber into 3-dimensional and 1-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha * dst[i,l,j] * src[l]
 *
 * @param[in] alpha: Scalar factor
 * @param[in] src: Input fiber, that is reshaped into 1D array
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    multiply_fiber_inplace_async<T>(alpha, src, dst, axis);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void multiply_fiber_inplace_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst, Index axis);

template
void multiply_fiber_inplace_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst, Index axis);

template
void multiply_fiber_inplace_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst, Index axis);

template
void multiply_fiber_inplace_async<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src,
        const Tensor<fp32_fast_fp16_t> &dst, Index axis);

template
void multiply_fiber_inplace_async<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src,
        const Tensor<fp32_fast_bf16_t> &dst, Index axis);

template
void multiply_fiber_inplace_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst, Index axis);

// Explicit instantiation of template
template
void multiply_fiber_inplace<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst, Index axis);

template
void multiply_fiber_inplace<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst, Index axis);

template
void multiply_fiber_inplace<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst, Index axis);

template
void multiply_fiber_inplace<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src,
        const Tensor<fp32_fast_fp16_t> &dst, Index axis);

template
void multiply_fiber_inplace<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src,
        const Tensor<fp32_fast_bf16_t> &dst, Index axis);

template
void multiply_fiber_inplace<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst, Index axis);

} // namespace nntile::tensor
