/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/multiply_inplace.cc
 * Per-element product of two Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/multiply_inplace.hh"
#include "nntile/tile/multiply_inplace.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise prod operation
/*! @param[in] alpha: Scalar multiplier
 * @param[in] src: Input tensor for the prod operation
 * @param[inout] dst: Input and output tensor for the prod operation
 * */
template<typename T>
void multiply_inplace_async(Scalar alpha, const Tensor<T> &src, const Tensor<T> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    // Check shapes of base tiles
    if(src.basetile_shape != dst.basetile_shape)
    {
        throw std::runtime_error("src.basetile_shape != dst.basetile_shape");
    }
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        auto dst_tile_handle = dst.get_tile_handle(i);
        auto src_tile = src.get_tile(i);
        auto dst_tile = dst.get_tile(i);
        tile::multiply_inplace_async<T>(alpha, src_tile, dst_tile);
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise prod operation
/*! @param[in] alpha: Scalar multiplier
 * @param[in] src: Input tensor for the prod operation
 * @param[inout] dst: Input and output tensor for the prod operation
 * */
template<typename T>
void multiply_inplace(Scalar alpha, const Tensor<T> &src, const Tensor<T> &dst)
{
    multiply_inplace_async<T>(alpha, src, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void multiply_inplace_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void multiply_inplace_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void multiply_inplace_async<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src,
        const Tensor<fp32_fast_fp16_t> &dst);

template
void multiply_inplace_async<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src,
        const Tensor<fp32_fast_bf16_t> &dst);

template
void multiply_inplace_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

template
void multiply_inplace_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst);

template
void multiply_inplace_async<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src,
        const Tensor<fp16_t> &dst);

// Explicit instantiation
template
void multiply_inplace<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void multiply_inplace<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void multiply_inplace<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src,
        const Tensor<fp32_fast_fp16_t> &dst);

template
void multiply_inplace<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src,
        const Tensor<fp32_fast_bf16_t> &dst);

template
void multiply_inplace<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

template
void multiply_inplace<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst);

template
void multiply_inplace<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src,
        const Tensor<fp16_t> &dst);

} // namespace nntile::tensor
