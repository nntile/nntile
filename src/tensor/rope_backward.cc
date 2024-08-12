/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/rope_backward.cc
 * Backward RoPE operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/rope_backward.hh"
#include "nntile/starpu/rope_backward.hh"
#include <iostream>

namespace nntile::tensor
{

template<typename T>
void rope_backward_async(const Tensor<T> &sin, const Tensor<T> &cos,
        const Tensor<T> &dy, const Tensor<T> &dx)
//! Tensor<T> Backward for Rotary Positional Embedding
/*!
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] dy: Input embedding tensor
 * @param[inout] dx: Output embedding tensor with applied RoPE
 * */
{
    // Check dimensions
    if(dy.ndim != dx.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }

    if(sin.ndim != cos.ndim)
    {
        throw std::runtime_error("sin.ndim != cos.ndim");
    }

    if(dy.ndim < sin.ndim)
    {
        throw std::runtime_error("src.ndim < sin.ndim");
    }

    if(dy.shape != dx.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }

    if(dy.basetile_shape != dx.basetile_shape)
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
    if(dx.shape[0] != 2*sin.shape[0])
    {
        throw std::runtime_error("src.shape[0] != 2*sin.shape[0]");
    }

    if(dx.basetile_shape[0] != 2*sin.basetile_shape[0])
    {
        throw std::runtime_error("src.basetile_shape[0] != 2*sin.basetile_shape[0]");
    }

    for(Index i = 1; i < sin.ndim; ++i)
    {
        if(dy.shape[i] != sin.shape[i])
        {
            throw std::runtime_error("src.shape[i] != sin.shape[i]");
        }
    }

    for(Index i = 1; i < sin.ndim; ++i)
    {
        if(dy.basetile_shape[i] != sin.basetile_shape[i])
        {
            throw std::runtime_error("src.basetile_shape[i] != sin.basetile_shape[i]");
        }
    }

    // Apply per-tile rope asynchronously
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    for(Index i = 0; i < dy.grid.nelems; ++i)
    {
        // Index of current source and destination tiles
        auto dydx_tile_index = dy.grid.linear_to_index(i);

        // Index of sin and cos tiles
        std::vector<Index> sincos_tile_index(dydx_tile_index.cbegin(),
                dydx_tile_index.cbegin()+sin.ndim);
        Index j = sin.grid.index_to_linear(sincos_tile_index);

        // Get all tile handles
        auto dy_tile_handle = dy.get_tile_handle(i);
        auto dx_tile_handle = dx.get_tile_handle(i);
        auto sin_tile_handle = sin.get_tile_handle(j);
        auto cos_tile_handle = cos.get_tile_handle(j);

        // Get tile traits
        auto dydx_tile_traits = dy.get_tile_traits(i);
        auto sincos_tile_traits = sin.get_tile_traits(j);

        // Sizes of underlying StarPU submit call
        Index m{sincos_tile_traits.nelems},
              n{dydx_tile_traits.matrix_shape[sin.ndim][1]};

        // Insert corresponding task
        starpu::rope_backward::submit<T>(m, n, sin_tile_handle, cos_tile_handle,
                dy_tile_handle, dx_tile_handle);
    }
}

template<typename T>
void rope_backward(const Tensor<T> &sin, const Tensor<T> &cos, const Tensor<T> &dy,
        const Tensor<T> &dx)
//!
/*! Blocking version of rope_async<T>.
 *
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] dy: Input embedding tensor
 * @param[inout] dx: Output embedding tensor with applied RoPE
 * */
{
    rope_backward_async<T>(sin, cos, dy, dx);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void rope_backward_async<fp32_t>(const Tensor<fp32_t> &sin, const Tensor<fp32_t> &cos,
        const Tensor<fp32_t> &dy, const Tensor<fp32_t> &dx);

template
void rope_backward_async<fp64_t>(const Tensor<fp64_t> &sin, const Tensor<fp64_t> &cos,
        const Tensor<fp64_t> &dy, const Tensor<fp64_t> &dx);

template
void rope_backward_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &sin,
        const Tensor<fp32_fast_tf32_t> &cos,
        const Tensor<fp32_fast_tf32_t> &dy,
        const Tensor<fp32_fast_tf32_t> &dx);

template
void rope_backward_async<bf16_t>(const Tensor<bf16_t> &sin, const Tensor<bf16_t> &cos,
        const Tensor<bf16_t> &dy, const Tensor<bf16_t> &dx);

// Explicit instantiation of template
template
void rope_backward<fp32_t>(const Tensor<fp32_t> &sin, const Tensor<fp32_t> &cos,
        const Tensor<fp32_t> &dy, const Tensor<fp32_t> &dx);

template
void rope_backward<fp64_t>(const Tensor<fp64_t> &sin, const Tensor<fp64_t> &cos,
        const Tensor<fp64_t> &dy, const Tensor<fp64_t> &dx);

template
void rope_backward<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &sin,
        const Tensor<fp32_fast_tf32_t> &cos,
        const Tensor<fp32_fast_tf32_t> &dy,
        const Tensor<fp32_fast_tf32_t> &dx);

template
void rope_backward<bf16_t>(const Tensor<bf16_t> &sin, const Tensor<bf16_t> &cos,
        const Tensor<bf16_t> &dy, const Tensor<bf16_t> &dx);

} // namespace tensor
