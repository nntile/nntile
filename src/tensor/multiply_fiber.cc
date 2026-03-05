/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/multiply_fiber.cc
 * Tensor wrappers for per-element product of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/multiply_fiber.hh"
#include "nntile/tile/multiply_fiber.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

template<typename T>
void multiply_fiber_async(Scalar alpha, const Tensor<T> &src1, const Tensor<T> &src2,
        const Tensor<T> &dst, Index axis)
//! Tensor<T> per-element multiplication of a tensor and a broadcasted fiber
/*! Reshapes input tensor and fiber into 3-dimensional and 1-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha * src1[l] * src2[i,l,j]
 *
 * @param[in] alpha: Scalar factor
 * @param[in] src: Input fiber, that is reshaped into 1D array
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    // Check dimensions
    if(src1.ndim != 1)
    {
        throw std::runtime_error("src1.ndim != 1");
    }
    if(src2.ndim != dst.ndim)
    {
        throw std::runtime_error("src2.ndim != dst.ndim");
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
    if(src1.shape[0] != dst.shape[axis])
    {
        throw std::runtime_error("src1.shape[0] != dst.shape[axis]");
    }
    if(src1.basetile_shape[0] != dst.basetile_shape[axis])
    {
        throw std::runtime_error("src1.basetile_shape[0] != "
                "dst.basetile_shape[axis]");
    }
    if(src2.shape != dst.shape)
    {
        throw std::runtime_error("src2.shape != dst.shape");
    }
    if(src2.basetile_shape != dst.basetile_shape)
    {
        throw std::runtime_error("src2.basetile_shape != dst.basetile_shape");
    }
    // Apply per-tile multiply_fiber asynchronously as needed
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        auto dst_tile_index = dst.grid.linear_to_index(i);
        auto dst_tile_handle = dst.get_tile_handle(i);
        auto dst_tile = dst.get_tile(i);
        // Get corresponding src tile
        Index j = dst_tile_index[axis];
        auto src1_tile = src1.get_tile(j);
        auto src2_tile = src2.get_tile(i);
        tile::multiply_fiber_async<T>(alpha, src1_tile, src2_tile, dst_tile,
                axis);
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

template<typename T>
void multiply_fiber(Scalar alpha, const Tensor<T> &src1, const Tensor<T> &src2,
        const Tensor<T> &dst, Index axis)
//! Tensor<T> per-element multiplication of a tensor and a broadcasted fiber
/*! Blocking version of multiply_fiber_async<T>.
 * Reshapes input tensor and fiber into 3-dimensional and 1-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha * src1[l] * src2[i,l,j]
 *
 * @param[in] alpha: Scalar factor
 * @param[in] src: Input fiber, that is reshaped into 1D array
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    multiply_fiber_async<T>(alpha, src1, src2, dst, axis);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void multiply_fiber_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src1,
        const Tensor<fp32_t> &src2, const Tensor<fp32_t> &dst, Index axis);

template
void multiply_fiber_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src1,
        const Tensor<fp32_fast_tf32_t> &src2, const Tensor<fp32_fast_tf32_t> &dst, Index axis);

template
void multiply_fiber_async<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src1,
        const Tensor<fp32_fast_fp16_t> &src2, const Tensor<fp32_fast_fp16_t> &dst, Index axis);

template
void multiply_fiber_async<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src1,
        const Tensor<fp32_fast_bf16_t> &src2, const Tensor<fp32_fast_bf16_t> &dst, Index axis);

template
void multiply_fiber_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src1,
        const Tensor<fp64_t> &src2, const Tensor<fp64_t> &dst, Index axis);

template
void multiply_fiber_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src1,
        const Tensor<bf16_t> &src2, const Tensor<bf16_t> &dst, Index axis);

template
void multiply_fiber_async<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src1,
        const Tensor<fp16_t> &src2, const Tensor<fp16_t> &dst, Index axis);

// Explicit instantiation of template
template
void multiply_fiber<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src1,
        const Tensor<fp32_t> &src2, const Tensor<fp32_t> &dst, Index axis);

template
void multiply_fiber<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src1,
        const Tensor<fp32_fast_tf32_t> &src2, const Tensor<fp32_fast_tf32_t> &dst, Index axis);

template
void multiply_fiber<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src1,
        const Tensor<fp32_fast_fp16_t> &src2, const Tensor<fp32_fast_fp16_t> &dst, Index axis);

template
void multiply_fiber<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src1,
        const Tensor<fp32_fast_bf16_t> &src2, const Tensor<fp32_fast_bf16_t> &dst, Index axis);

template
void multiply_fiber<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src1,
        const Tensor<fp64_t> &src2, const Tensor<fp64_t> &dst, Index axis);

template
void multiply_fiber<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src1,
        const Tensor<bf16_t> &src2, const Tensor<bf16_t> &dst, Index axis);

template
void multiply_fiber<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src1,
        const Tensor<fp16_t> &src2, const Tensor<fp16_t> &dst, Index axis);

} // namespace nntile::tensor
