/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/rope_backward.cc
 * Backward RoPE operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/rope_backward.hh"
#include "nntile/starpu/rope_backward.hh"
#include "nntile/starpu/config.hh"

namespace nntile::core
{

template<typename T>
void rope_backward_async(int starpu_worker_hint, const Tile<T> &sin, const Tile<T> &cos,
        const Tile<T> &dy, const Tile<T> &dx, Index sin_pair0)
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
    const Index rope_axis = sin.ndim - 1;
    if(dy.shape[rope_axis] != 2 * sin.shape[rope_axis])
    {
        throw std::runtime_error(
            "dy.shape[rope_axis] != 2*sin.shape[rope_axis]");
    }
    for(Index i = 0; i < rope_axis; ++i)
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
        const Index nrows = dy.matrix_shape[sin.ndim][1];
        const Index ncols = sin.nelems;
        starpu::rope_backward.submit<std::tuple<T>>(starpu_worker_hint, nrows,
            ncols, sin_pair0, sin, cos, dy, dx);
    }
}

template<typename T>
void rope_backward(int starpu_worker_hint, const Tile<T> &sin, const Tile<T> &cos, const Tile<T> &dy,
        const Tile<T> &dx, Index sin_pair0)
{
    rope_backward_async<T>(starpu_worker_hint, sin, cos, dy, dx, sin_pair0);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void rope_backward_async<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &sin,
        const Tile<fp32_t> &cos, const Tile<fp32_t> &dy,
        const Tile<fp32_t> &dx, Index sin_pair0);

template
void rope_backward_async<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &sin,
        const Tile<fp64_t> &cos, const Tile<fp64_t> &dy,
        const Tile<fp64_t> &dx, Index sin_pair0);

template
void rope_backward_async<fp32_fast_tf32_t>(int starpu_worker_hint, 
        const Tile<fp32_fast_tf32_t> &sin,
        const Tile<fp32_fast_tf32_t> &cos,
        const Tile<fp32_fast_tf32_t> &dy,
        const Tile<fp32_fast_tf32_t> &dx, Index sin_pair0);

template
void rope_backward_async<fp32_fast_fp16_t>(int starpu_worker_hint, 
        const Tile<fp32_fast_fp16_t> &sin,
        const Tile<fp32_fast_fp16_t> &cos,
        const Tile<fp32_fast_fp16_t> &dy,
        const Tile<fp32_fast_fp16_t> &dx, Index sin_pair0);

template
void rope_backward_async<fp32_fast_bf16_t>(int starpu_worker_hint, 
        const Tile<fp32_fast_bf16_t> &sin,
        const Tile<fp32_fast_bf16_t> &cos,
        const Tile<fp32_fast_bf16_t> &dy,
        const Tile<fp32_fast_bf16_t> &dx, Index sin_pair0);

template
void rope_backward_async<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &sin,
        const Tile<fp16_t> &cos, const Tile<fp16_t> &dy,
        const Tile<fp16_t> &dx, Index sin_pair0);

template
void rope_backward_async<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &sin,
        const Tile<bf16_t> &cos, const Tile<bf16_t> &dy,
        const Tile<bf16_t> &dx, Index sin_pair0);

// Explicit instantiation of template
template
void rope_backward<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &sin, const Tile<fp32_t> &cos,
        const Tile<fp32_t> &dy, const Tile<fp32_t> &dx, Index sin_pair0);

template
void rope_backward<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &sin, const Tile<fp64_t> &cos,
        const Tile<fp64_t> &dy, const Tile<fp64_t> &dx, Index sin_pair0);

template
void rope_backward<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<fp32_fast_tf32_t> &sin,
        const Tile<fp32_fast_tf32_t> &cos,
        const Tile<fp32_fast_tf32_t> &dy,
        const Tile<fp32_fast_tf32_t> &dx, Index sin_pair0);

template
void rope_backward<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<fp32_fast_fp16_t> &sin,
        const Tile<fp32_fast_fp16_t> &cos,
        const Tile<fp32_fast_fp16_t> &dy,
        const Tile<fp32_fast_fp16_t> &dx, Index sin_pair0);

template
void rope_backward<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<fp32_fast_bf16_t> &sin,
        const Tile<fp32_fast_bf16_t> &cos,
        const Tile<fp32_fast_bf16_t> &dy,
        const Tile<fp32_fast_bf16_t> &dx, Index sin_pair0);

template
void rope_backward<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &sin, const Tile<fp16_t> &cos,
        const Tile<fp16_t> &dy, const Tile<fp16_t> &dx, Index sin_pair0);

template
void rope_backward<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &sin, const Tile<bf16_t> &cos,
        const Tile<bf16_t> &dy, const Tile<bf16_t> &dx, Index sin_pair0);

} // namespace nntile::core
