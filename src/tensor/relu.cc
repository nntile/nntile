/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/relu.cc
 * ReLU operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/relu.hh"
#include "nntile/tile/relu.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

template<typename T>
void relu_async(const Tensor<T> &src, const Tensor<T> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    if(src.basetile_shape != dst.basetile_shape)
    {
        throw std::runtime_error("src.basetile_shape != dst.basetile_shape");
    }
    // Launch necessary tasks
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        auto dst_tile_handle = dst.get_tile_handle(i);
        auto src_tile = src.get_tile(i);
        auto dst_tile = dst.get_tile(i);
        tile::relu_async<T>(src_tile, dst_tile);
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

template<typename T>
void relu(const Tensor<T> &src, const Tensor<T> &dst)
{
    relu_async<T>(src, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void relu_async<fp32_t>(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void relu_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void relu_async<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &src,
        const Tensor<fp32_fast_fp16_t> &dst);

template
void relu_async<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &src,
        const Tensor<fp32_fast_bf16_t> &dst);

template
void relu_async<fp64_t>(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

template
void relu_async<bf16_t>(const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst);

template
void relu_async<fp16_t>(const Tensor<fp16_t> &src,
        const Tensor<fp16_t> &dst);

// Explicit instantiation
template
void relu<fp32_t>(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void relu<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void relu<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &src,
        const Tensor<fp32_fast_fp16_t> &dst);

template
void relu<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &src,
        const Tensor<fp32_fast_bf16_t> &dst);

template
void relu<fp64_t>(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

template
void relu<bf16_t>(const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst);

template
void relu<fp16_t>(const Tensor<fp16_t> &src,
        const Tensor<fp16_t> &dst);

} // namespace nntile::tensor
