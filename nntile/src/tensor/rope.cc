/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/rope.cc
 * Tensor wrappers for the Rotary Positional Embedding
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/rope.hh"
#include "nntile/tile/rope.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

template<typename T>
void rope_async(const Tensor<T> &sin, const Tensor<T> &cos,
        const Tensor<T> &src, const Tensor<T> &dst)
//! Tensor<T> Rotary Positional Embedding
/*!
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] src: Input embedding tensor
 * @param[out] dst: Output embedding tensor with applied RoPE
 * */
{
    // Check dimensions
    if(dst.ndim != src.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }

    if(sin.ndim != cos.ndim)
    {
        throw std::runtime_error("sin.ndim != cos.ndim");
    }

    if(src.ndim < sin.ndim)
    {
        throw std::runtime_error("src.ndim < sin.ndim");
    }

    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }

    if(src.basetile_shape != dst.basetile_shape)
    {
        throw std::runtime_error("src.basetile_shape != dst.basetile_shape");
    }

    if(sin.shape != cos.shape)
    {
        throw std::runtime_error("sin.shape != cos.shape");
    }

    if(sin.basetile_shape != cos.basetile_shape)
    {
        throw std::runtime_error("sin.basetile_shape != cos.basetile_shape");
    }

    if(sin.ndim == 0)
    {
        throw std::runtime_error("sin.ndim == 0");
    }

    // 0-th dimension is the head_size, which is halved for sin and cos
    if(src.shape[0] != 2*sin.shape[0])
    {
        throw std::runtime_error("src.shape[0] != 2*sin.shape[0]");
    }

    if(src.basetile_shape[0] != 2*sin.basetile_shape[0])
    {
        throw std::runtime_error("src.basetile_shape[0] != 2*sin.basetile_shape[0]");
    }

    for(Index i = 1; i < sin.ndim; ++i)
    {
        if(src.shape[i] != sin.shape[i])
        {
            throw std::runtime_error("src.shape[i] != sin.shape[i]");
        }
    }

    for(Index i = 1; i < sin.ndim; ++i)
    {
        if(src.basetile_shape[i] != sin.basetile_shape[i])
        {
            throw std::runtime_error("src.basetile_shape[i] != sin.basetile_shape[i]");
        }
    }

    // Apply per-tile rope asynchronously
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        // Index of current source and destination tiles
        auto srcdst_tile_index = src.grid.linear_to_index(i);

        // Index of sin and cos tiles
        std::vector<Index> sincos_tile_index(srcdst_tile_index.cbegin(),
                srcdst_tile_index.cbegin()+sin.ndim);
        Index j = sin.grid.index_to_linear(sincos_tile_index);

        // Get all tiles
        auto src_tile = src.get_tile(i);
        auto dst_tile = dst.get_tile(i);
        auto sin_tile = sin.get_tile(j);
        auto cos_tile = cos.get_tile(j);
        // Dispatch through tile wrapper
        tile::rope_async<T>(sin_tile, cos_tile, src_tile, dst_tile);
        // Flush cache for the output tile on every node
        dst.get_tile_handle(i).mpi_flush();
    }
}

template<typename T>
void rope(const Tensor<T> &sin, const Tensor<T> &cos, const Tensor<T> &src,
        const Tensor<T> &dst)
//!
/*! Blocking version of rope_async<T>.
 *
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] src: Input embedding tensor
 * @param[out] dst: Output embedding tensor with applied RoPE
 * */
{
    rope_async<T>(sin, cos, src, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void rope_async<fp32_t>(const Tensor<fp32_t> &sin, const Tensor<fp32_t> &cos,
        const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst);

template
void rope_async<fp64_t>(const Tensor<fp64_t> &sin, const Tensor<fp64_t> &cos,
        const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst);

template
void rope_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &sin,
        const Tensor<fp32_fast_tf32_t> &cos,
        const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void rope_async<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &sin,
        const Tensor<fp32_fast_fp16_t> &cos,
        const Tensor<fp32_fast_fp16_t> &src,
        const Tensor<fp32_fast_fp16_t> &dst);

template
void rope_async<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &sin,
        const Tensor<fp32_fast_bf16_t> &cos,
        const Tensor<fp32_fast_bf16_t> &src,
        const Tensor<fp32_fast_bf16_t> &dst);

template
void rope_async<fp16_t>(const Tensor<fp16_t> &sin, const Tensor<fp16_t> &cos,
        const Tensor<fp16_t> &src, const Tensor<fp16_t> &dst);

template
void rope_async<bf16_t>(const Tensor<bf16_t> &sin, const Tensor<bf16_t> &cos,
        const Tensor<bf16_t> &src, const Tensor<bf16_t> &dst);

// Explicit instantiation of template
template
void rope<fp32_t>(const Tensor<fp32_t> &sin, const Tensor<fp32_t> &cos,
        const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst);

template
void rope<fp64_t>(const Tensor<fp64_t> &sin, const Tensor<fp64_t> &cos,
        const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst);

template
void rope<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &sin,
        const Tensor<fp32_fast_tf32_t> &cos,
        const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void rope<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &sin,
        const Tensor<fp32_fast_fp16_t> &cos,
        const Tensor<fp32_fast_fp16_t> &src,
        const Tensor<fp32_fast_fp16_t> &dst);

template
void rope<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &sin,
        const Tensor<fp32_fast_bf16_t> &cos,
        const Tensor<fp32_fast_bf16_t> &src,
        const Tensor<fp32_fast_bf16_t> &dst);

template
void rope<fp16_t>(const Tensor<fp16_t> &sin, const Tensor<fp16_t> &cos,
        const Tensor<fp16_t> &src, const Tensor<fp16_t> &dst);

template
void rope<bf16_t>(const Tensor<bf16_t> &sin, const Tensor<bf16_t> &cos,
        const Tensor<bf16_t> &src, const Tensor<bf16_t> &dst);

} // namespace tensor
