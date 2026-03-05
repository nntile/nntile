/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/relu_backward.cc
 * Backward ReLU operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/relu_backward.hh"
#include "nntile/tile/relu_backward.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise backward relu operation
//
// @param[inout] A: Tensor for the element-wise backward relu operation
template<typename T>
void relu_backward_async(const Tensor<T> &x, const Tensor<T> &dy,
        const Tensor<T> &dx)
{
    // Check shapes
    if(x.shape != dy.shape)
    {
        throw std::runtime_error("x.shape != dy.shape");
    }
    if(x.basetile_shape != dy.basetile_shape)
    {
        throw std::runtime_error("x.basetile_shape != dy.basetile_shape");
    }
    if(x.shape != dx.shape)
    {
        throw std::runtime_error("x.shape != dx.shape");
    }
    if(x.basetile_shape != dx.basetile_shape)
    {
        throw std::runtime_error("x.basetile_shape != dx.basetile_shape");
    }
    // Do actual calculations
    for(Index i = 0; i < x.grid.nelems; ++i)
    {
        auto dx_tile_handle = dx.get_tile_handle(i);
        auto x_tile = x.get_tile(i);
        auto dy_tile = dy.get_tile(i);
        auto dx_tile = dx.get_tile(i);
        tile::relu_backward_async<T>(x_tile, dy_tile, dx_tile);
        // Clear cached output value
        dx_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise backward relu operation
//
// @param[inout] A: Tensor for the element-wise backward relu operation
template<typename T>
void relu_backward(const Tensor<T> &x, const Tensor<T> &dy,
        const Tensor<T> &dx)
{
    relu_backward_async<T>(x, dy, dx);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void relu_backward_async<fp32_t>(const Tensor<fp32_t> &x,
        const Tensor<fp32_t> &dy, const Tensor<fp32_t> &dx);

template
void relu_backward_async<bf16_t>(const Tensor<bf16_t> &x,
        const Tensor<bf16_t> &dy, const Tensor<bf16_t> &dx);

template
void relu_backward_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &x,
        const Tensor<fp32_fast_tf32_t> &dy, const Tensor<fp32_fast_tf32_t> &dx);

template
void relu_backward_async<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &x,
        const Tensor<fp32_fast_fp16_t> &dy, const Tensor<fp32_fast_fp16_t> &dx);

template
void relu_backward_async<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &x,
        const Tensor<fp32_fast_bf16_t> &dy, const Tensor<fp32_fast_bf16_t> &dx);

template
void relu_backward_async<fp64_t>(const Tensor<fp64_t> &x,
        const Tensor<fp64_t> &dy, const Tensor<fp64_t> &dx);

// Explicit instantiation
template
void relu_backward<fp32_t>(const Tensor<fp32_t> &x,
        const Tensor<fp32_t> &dy, const Tensor<fp32_t> &dx);

template
void relu_backward<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &x,
        const Tensor<fp32_fast_tf32_t> &dy, const Tensor<fp32_fast_tf32_t> &dx);

template
void relu_backward<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &x,
        const Tensor<fp32_fast_fp16_t> &dy, const Tensor<fp32_fast_fp16_t> &dx);

template
void relu_backward<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &x,
        const Tensor<fp32_fast_bf16_t> &dy, const Tensor<fp32_fast_bf16_t> &dx);

template
void relu_backward<fp64_t>(const Tensor<fp64_t> &x,
        const Tensor<fp64_t> &dy, const Tensor<fp64_t> &dx);

template
void relu_backward<bf16_t>(const Tensor<bf16_t> &x,
        const Tensor<bf16_t> &dy, const Tensor<bf16_t> &dx);

} // namespace nntile::tensor
