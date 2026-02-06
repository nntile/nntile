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
#include "nntile/starpu/gelu.hh"
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
    int mpi_rank = starpu_mpi_world_rank();
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        // Get handle for corresponding tiles of src and dst
        auto src_tile_handle = src.get_tile_handle(i);
        auto dst_tile_handle = dst.get_tile_handle(i);
        // MPI rank of the destination tile
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Transfer data
        src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        // Execute only on destination node
        if(mpi_rank == dst_tile_rank)
        {
            auto tile_traits = src.get_tile_traits(i);
            starpu::gelu.submit<std::tuple<T>>(tile_traits.nelems, src_tile_handle,
                    dst_tile_handle);
        }
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
