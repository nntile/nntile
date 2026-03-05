/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/rope_backward.cc
 * Backward RoPE operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/rope_backward.hh"
#include "nntile/starpu/rope_backward.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

template<typename T>
void rope_backward_async(const Tile<T> &sin, const Tile<T> &cos,
        const Tile<T> &dy, const Tile<T> &dx)
{
    // Check dimensions
    if(dy.ndim != dx.ndim)
    {
        throw std::runtime_error("dx.ndim != dy.ndim");
    }
    if(sin.ndim != cos.ndim)
    {
        throw std::runtime_error("sin.ndim != cos.ndim");
    }
    if(dy.ndim < sin.ndim)
    {
        throw std::runtime_error("dy.ndim < sin.ndim");
    }
    if(dy.shape != dx.shape)
    {
        throw std::runtime_error("dy.shape != dx.shape");
    }
    if(sin.shape != cos.shape)
    {
        throw std::runtime_error("sin.shape != cos.shape");
    }
    if(sin.ndim == 0)
    {
        throw std::runtime_error("sin.ndim == 0");
    }
    // 0-th dimension is the head_size, which is halved for sin and cos
    if(dy.shape[0] != 2*sin.shape[0])
    {
        throw std::runtime_error("dy.shape[0] != 2*sin.shape[0]");
    }
    for(Index i = 1; i < sin.ndim; ++i)
    {
        if(dy.shape[i] != sin.shape[i])
        {
            throw std::runtime_error("dy.shape[i] != sin.shape[i]");
        }
    }

    int mpi_rank = starpu_mpi_world_rank();
    int dx_rank = dx.mpi_get_rank();
    sin.mpi_transfer(dx_rank, mpi_rank);
    cos.mpi_transfer(dx_rank, mpi_rank);
    dy.mpi_transfer(dx_rank, mpi_rank);
    if(mpi_rank == dx_rank)
    {
        Index m{sin.nelems};
        Index n{dy.matrix_shape[sin.ndim][1]};
        starpu::rope_backward.submit<std::tuple<T>>(m, n, sin, cos, dy, dx);
    }
}

template<typename T>
void rope_backward(const Tile<T> &sin, const Tile<T> &cos, const Tile<T> &dy,
        const Tile<T> &dx)
{
    rope_backward_async<T>(sin, cos, dy, dx);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void rope_backward_async<fp32_t>(const Tile<fp32_t> &sin,
        const Tile<fp32_t> &cos, const Tile<fp32_t> &dy,
        const Tile<fp32_t> &dx);

template
void rope_backward_async<fp64_t>(const Tile<fp64_t> &sin,
        const Tile<fp64_t> &cos, const Tile<fp64_t> &dy,
        const Tile<fp64_t> &dx);

template
void rope_backward_async<fp32_fast_tf32_t>(
        const Tile<fp32_fast_tf32_t> &sin,
        const Tile<fp32_fast_tf32_t> &cos,
        const Tile<fp32_fast_tf32_t> &dy,
        const Tile<fp32_fast_tf32_t> &dx);

template
void rope_backward_async<fp32_fast_fp16_t>(
        const Tile<fp32_fast_fp16_t> &sin,
        const Tile<fp32_fast_fp16_t> &cos,
        const Tile<fp32_fast_fp16_t> &dy,
        const Tile<fp32_fast_fp16_t> &dx);

template
void rope_backward_async<fp32_fast_bf16_t>(
        const Tile<fp32_fast_bf16_t> &sin,
        const Tile<fp32_fast_bf16_t> &cos,
        const Tile<fp32_fast_bf16_t> &dy,
        const Tile<fp32_fast_bf16_t> &dx);

template
void rope_backward_async<fp16_t>(const Tile<fp16_t> &sin,
        const Tile<fp16_t> &cos, const Tile<fp16_t> &dy,
        const Tile<fp16_t> &dx);

template
void rope_backward_async<bf16_t>(const Tile<bf16_t> &sin,
        const Tile<bf16_t> &cos, const Tile<bf16_t> &dy,
        const Tile<bf16_t> &dx);

// Explicit instantiation of template
template
void rope_backward<fp32_t>(const Tile<fp32_t> &sin, const Tile<fp32_t> &cos,
        const Tile<fp32_t> &dy, const Tile<fp32_t> &dx);

template
void rope_backward<fp64_t>(const Tile<fp64_t> &sin, const Tile<fp64_t> &cos,
        const Tile<fp64_t> &dy, const Tile<fp64_t> &dx);

template
void rope_backward<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &sin,
        const Tile<fp32_fast_tf32_t> &cos,
        const Tile<fp32_fast_tf32_t> &dy,
        const Tile<fp32_fast_tf32_t> &dx);

template
void rope_backward<fp32_fast_fp16_t>(const Tile<fp32_fast_fp16_t> &sin,
        const Tile<fp32_fast_fp16_t> &cos,
        const Tile<fp32_fast_fp16_t> &dy,
        const Tile<fp32_fast_fp16_t> &dx);

template
void rope_backward<fp32_fast_bf16_t>(const Tile<fp32_fast_bf16_t> &sin,
        const Tile<fp32_fast_bf16_t> &cos,
        const Tile<fp32_fast_bf16_t> &dy,
        const Tile<fp32_fast_bf16_t> &dx);

template
void rope_backward<fp16_t>(const Tile<fp16_t> &sin, const Tile<fp16_t> &cos,
        const Tile<fp16_t> &dy, const Tile<fp16_t> &dx);

template
void rope_backward<bf16_t>(const Tile<bf16_t> &sin, const Tile<bf16_t> &cos,
        const Tile<bf16_t> &dy, const Tile<bf16_t> &dx);

} // namespace nntile::tile
