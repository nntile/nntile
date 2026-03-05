/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/gelu.cc
 * GeLU operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/gelu.hh"
#include "nntile/tile/gelu.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise GeLU operation
//
// @param[in] src: Input tensor for the element-wise GeLU operation
// @param[out] dst: Output tensor for the element-wise GeLU operation
template<typename T>
void gelu_async(const Tensor<T> &src, const Tensor<T> &dst)
{
    // Check dimensions
    if(dst.ndim != src.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }
    // Check shapes of tensors
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
        if(dst.basetile_shape[i] != src.basetile_shape[i])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "src.basetile_shape[i]");
        }
    }
    // Apply per-tile gelu asynchronously as needed
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        auto dst_tile_handle = dst.get_tile_handle(i);
        auto src_tile = src.get_tile(i);
        auto dst_tile = dst.get_tile(i);
        tile::gelu_async<T>(src_tile, dst_tile);
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise GeLU operation
//
// @param[in] src: Input tensor for the element-wise GeLU operation
// @param[out] dst: Output tensor for the element-wise GeLU operation
template<typename T>
void gelu(const Tensor<T> &src, const Tensor<T> &dst)
{
    gelu_async<T>(src, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void gelu_async<fp32_t>(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void gelu_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void gelu_async<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &src,
                                const Tensor<fp32_fast_fp16_t> &dst);

template
void gelu_async<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &src,
                                const Tensor<fp32_fast_bf16_t> &dst);

template
void gelu_async<fp64_t>(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

template
void gelu_async<bf16_t>(const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst);

template
void gelu_async<fp16_t>(const Tensor<fp16_t> &src,
        const Tensor<fp16_t> &dst);

// Explicit instantiation
template
void gelu<fp32_t>(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void gelu<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void gelu<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &src,
        const Tensor<fp32_fast_fp16_t> &dst);

template
void gelu<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &src,
        const Tensor<fp32_fast_bf16_t> &dst);

template
void gelu<fp64_t>(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

template
void gelu<bf16_t>(const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst);

template
void gelu<fp16_t>(const Tensor<fp16_t> &src,
        const Tensor<fp16_t> &dst);

} // namespace nntile::tensor
