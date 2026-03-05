/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/multiply.cc
 * Per-element product of two Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/multiply.hh"
#include "nntile/tile/multiply.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise multiply operation
/*! @param[in] alpha: Scalar multiplier
 * @param[in] src1: Input tensor for the multiply operation
 * @param[in] src2: Input tensor for the multiply operation
 * @param[inout] dst: Input and output tensor for the multiply operation
 * */
template<typename T>
void multiply_async(Scalar alpha, const Tensor<T> &src1, const Tensor<T> &src2,
        const Tensor<T> &dst)
{
    // Check shapes
    if(src1.shape != src2.shape)
    {
        throw std::runtime_error("src1.shape != src2.shape");
    }
    if(src1.shape != dst.shape)
    {
        throw std::runtime_error("src1.shape != dst.shape");
    }
    // Check shapes of base tiles
    if(src1.basetile_shape != src2.basetile_shape)
    {
        throw std::runtime_error("src1.basetile_shape != src2.basetile_shape");
    }
    if(src1.basetile_shape != dst.basetile_shape)
    {
        throw std::runtime_error("src1.basetile_shape != dst.basetile_shape");
    }
    for(Index i = 0; i < src1.grid.nelems; ++i)
    {
        auto dst_tile_handle = dst.get_tile_handle(i);
        auto src1_tile = src1.get_tile(i);
        auto src2_tile = src2.get_tile(i);
        auto dst_tile = dst.get_tile(i);
        tile::multiply_async<T>(alpha, src1_tile, src2_tile, dst_tile);
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise multiply operation
/*! @param[in] alpha: Scalar multiplier
 * @param[in] src1: Input tensor for the multiply operation
 * @param[in] src2: Input tensor for the multiply operation
 * @param[inout] dst: Input and output tensor for the multiply operation
 * */
template<typename T>
void multiply(Scalar alpha, const Tensor<T> &src1, const Tensor<T> &src2, const Tensor<T> &dst)
{
    multiply_async<T>(alpha, src1, src2, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void multiply_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src1, const Tensor<fp32_t> &src2,
        const Tensor<fp32_t> &dst);

template
void multiply_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src1,
        const Tensor<fp32_fast_tf32_t> &src2,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void multiply_async<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src1,
        const Tensor<fp32_fast_fp16_t> &src2,
        const Tensor<fp32_fast_fp16_t> &dst);

template
void multiply_async<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src1,
        const Tensor<fp32_fast_bf16_t> &src2,
        const Tensor<fp32_fast_bf16_t> &dst);

template
void multiply_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src1, const Tensor<fp64_t> &src2,
        const Tensor<fp64_t> &dst);

template
void multiply_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src1, const Tensor<bf16_t> &src2,
        const Tensor<bf16_t> &dst);

template
void multiply_async<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src1, const Tensor<fp16_t> &src2,
        const Tensor<fp16_t> &dst);

// Explicit instantiation
template
void multiply<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src1, const Tensor<fp32_t> &src2,
        const Tensor<fp32_t> &dst);

template
void multiply<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src1,
        const Tensor<fp32_fast_tf32_t> &src2,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void multiply<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src1,
        const Tensor<fp32_fast_fp16_t> &src2,
        const Tensor<fp32_fast_fp16_t> &dst);

template
void multiply<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src1,
        const Tensor<fp32_fast_bf16_t> &src2,
        const Tensor<fp32_fast_bf16_t> &dst);

template
void multiply<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src1, const Tensor<fp64_t> &src2,
        const Tensor<fp64_t> &dst);

template
void multiply<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src1, const Tensor<bf16_t> &src2,
        const Tensor<bf16_t> &dst);

template
void multiply<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src1, const Tensor<fp16_t> &src2,
        const Tensor<fp16_t> &dst);

} // namespace nntile::tensor
