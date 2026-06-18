/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/rope.cc
 * Tile wrappers for the Rotary Positional Embedding
 *
 * @version 1.1.0
 * */

#include "nntile/core/rope.hh"
#include "nntile/starpu/rope.hh"
#include "nntile/starpu/config.hh"

namespace nntile::core
{

template<typename T>
void rope_async(int starpu_worker_hint, const Tile<T> &sin, const Tile<T> &cos, const Tile<T> &src,
        const Tile<T> &dst, Index sin_pair0)
//! Tile<T> Rotary Positional Embedding
/*! Reshapes input tensor and slice into 3-dimensional and 2-dimensional arrays
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

    if(sin.shape != cos.shape)
    {
        throw std::runtime_error("sin.shape != cos.shape");
    }

    if(sin.ndim == 0)
    {
        throw std::runtime_error("sin.ndim == 0");
    }

    const Index half_axis = sin.ndim - 1;
    const Index axis_shift = src.ndim - sin.ndim;
    if(axis_shift < 0)
    {
        throw std::runtime_error("src.ndim < sin.ndim");
    }
    if(src.shape[half_axis + axis_shift] != 2 * sin.shape[half_axis])
    {
        throw std::runtime_error(
            "src head axis != 2*sin half axis");
    }
    for(Index i = 0; i < half_axis; ++i)
    {
        if(src.shape[i + axis_shift] != sin.shape[i])
        {
            throw std::runtime_error("src/sin batch axis mismatch");
        }
    }

    Index nrows = 1;
    for(Index i = 0; i < axis_shift; ++i)
    {
        nrows *= src.shape[i];
    }
    const Index ncols = sin.nelems;
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    sin.mpi_transfer(dst_rank, mpi_rank);
    cos.mpi_transfer(dst_rank, mpi_rank);
    src.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        if(sin_pair0 != 0)
        {
            throw std::runtime_error("rope: sin_pair0 != 0 is not supported");
        }
        starpu::rope.submit<std::tuple<T>>(starpu_worker_hint, ncols, nrows,
            sin, cos, src, dst);
    }
}

template<typename T>
void rope(int starpu_worker_hint, const Tile<T> &sin, const Tile<T> &cos, const Tile<T> &src,
        const Tile<T> &dst, Index sin_pair0)
//! Tile<T> addition of a tensor and a broadcasted slice
/*! Blocking version of rope_async<T>.
 *
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] src: Input embedding tensor
 * @param[out] dst: Output embedding tensor with applied RoPE
 * */
{
    rope_async<T>(starpu_worker_hint, sin, cos, src, dst, sin_pair0);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void rope_async<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &sin, const Tile<fp32_t> &cos,
        const Tile<fp32_t> &src, const Tile<fp32_t> &dst, Index sin_pair0);

template
void rope_async<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &sin, const Tile<fp64_t> &cos,
        const Tile<fp64_t> &src, const Tile<fp64_t> &dst, Index sin_pair0);

template
void rope_async<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<fp32_fast_tf32_t> &sin,
        const Tile<fp32_fast_tf32_t> &cos,
        const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst, Index sin_pair0);

template
void rope_async<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<fp32_fast_fp16_t> &sin,
        const Tile<fp32_fast_fp16_t> &cos,
        const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst, Index sin_pair0);

template
void rope_async<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<fp32_fast_bf16_t> &sin,
        const Tile<fp32_fast_bf16_t> &cos,
        const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst, Index sin_pair0);

template
void rope_async<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &sin, const Tile<bf16_t> &cos,
        const Tile<bf16_t> &src, const Tile<bf16_t> &dst, Index sin_pair0);

template
void rope_async<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &sin, const Tile<fp16_t> &cos,
        const Tile<fp16_t> &src, const Tile<fp16_t> &dst, Index sin_pair0);

// Explicit instantiation of template
template
void rope<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &sin, const Tile<fp32_t> &cos,
        const Tile<fp32_t> &src, const Tile<fp32_t> &dst, Index sin_pair0);

template
void rope<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &sin, const Tile<fp64_t> &cos,
        const Tile<fp64_t> &src, const Tile<fp64_t> &dst, Index sin_pair0);

template
void rope<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<fp32_fast_tf32_t> &sin,
        const Tile<fp32_fast_tf32_t> &cos,
        const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst, Index sin_pair0);

template
void rope<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<fp32_fast_fp16_t> &sin,
        const Tile<fp32_fast_fp16_t> &cos,
        const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst, Index sin_pair0);

template
void rope<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<fp32_fast_bf16_t> &sin,
        const Tile<fp32_fast_bf16_t> &cos,
        const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst, Index sin_pair0);

template
void rope<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &sin, const Tile<fp16_t> &cos,
        const Tile<fp16_t> &src, const Tile<fp16_t> &dst, Index sin_pair0);

template
void rope<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &sin, const Tile<bf16_t> &cos,
        const Tile<bf16_t> &src, const Tile<bf16_t> &dst, Index sin_pair0);

} // namespace nntile::core
