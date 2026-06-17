/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/softmax_inplace.cc
 * softmax_inplace operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/softmax_inplace.hh"
#include "nntile/starpu/softmax_inplace.hh"
#include "nntile/starpu/config.hh"

namespace nntile::core
{

template<typename T>
void softmax_inplace_async(int starpu_worker_hint, const Tile<T> &maxsumexp, Scalar alpha,
        const Tile<T> &dst, Index axis)
{
    // Check dimensions
    if(maxsumexp.ndim != dst.ndim)
    {
        throw std::runtime_error("maxsumexp.ndim != dst.ndim");
    }
    // Input shape dimension shall be at least 1
    if(maxsumexp.ndim == 0)
    {
        throw std::runtime_error("maxsumexp.ndim == 0");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= dst.ndim)
    {
        throw std::runtime_error("axis >= dst.ndim");
    }
    // Check shapes (C-order trailing pair dim).
    if(maxsumexp.shape[maxsumexp.ndim-1] != 2)
    {
        throw std::runtime_error("maxsumexp last dim must be 2");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != maxsumexp.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != maxsumexp.shape[i]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != maxsumexp.shape[i-1])
        {
            throw std::runtime_error("dst.shape[i] != maxsumexp.shape[i-1]");
        }
    }
    // Reshape inputs for simplicity: maxsumexp -> (2,m,n), dst -> (m,k,n)
    // dst is a part of (m,l,n) tensor
    Index m, n, k;
    m = dst.matrix_shape[axis+1][1];
    n = dst.matrix_shape[axis][0];
    k = dst.shape[axis];
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    maxsumexp.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        // Insert task
        starpu::softmax_inplace.submit<std::tuple<T>>(starpu_worker_hint, m, n, k, maxsumexp,
                alpha, dst);
    }
}

//! Tile-wise average and deviation from sum and scaled sum of squares
template<typename T>
void softmax_inplace(int starpu_worker_hint, const Tile<T> &maxsumexp, Scalar alpha, const Tile<T> &dst,
        Index axis)
{
    softmax_inplace_async<T>(starpu_worker_hint, maxsumexp, alpha, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void softmax_inplace_async<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &maxsumexp, Scalar alpha,
        const Tile<fp32_t> &dst, Index axis);

template
void softmax_inplace_async<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<fp32_fast_tf32_t> &maxsumexp, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &dst, Index axis);

template
void softmax_inplace_async<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<fp32_fast_fp16_t> &maxsumexp, Scalar alpha,
        const Tile<fp32_fast_fp16_t> &dst, Index axis);

template
void softmax_inplace_async<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<fp32_fast_bf16_t> &maxsumexp, Scalar alpha,
        const Tile<fp32_fast_bf16_t> &dst, Index axis);

template
void softmax_inplace_async<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &maxsumexp, Scalar alpha,
        const Tile<fp64_t> &dst, Index axis);

template
void softmax_inplace_async<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &maxsumexp, Scalar alpha,
        const Tile<bf16_t> &dst, Index axis);

template
void softmax_inplace_async<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &maxsumexp, Scalar alpha,
        const Tile<fp16_t> &dst, Index axis);

// Explicit instantiation
template
void softmax_inplace<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &maxsumexp, Scalar alpha,
        const Tile<fp32_t> &dst, Index axis);

template
void softmax_inplace<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<fp32_fast_tf32_t> &maxsumexp, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &dst, Index axis);

template
void softmax_inplace<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<fp32_fast_fp16_t> &maxsumexp, Scalar alpha,
        const Tile<fp32_fast_fp16_t> &dst, Index axis);

template
void softmax_inplace<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<fp32_fast_bf16_t> &maxsumexp, Scalar alpha,
        const Tile<fp32_fast_bf16_t> &dst, Index axis);

template
void softmax_inplace<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &maxsumexp, Scalar alpha,
        const Tile<fp64_t> &dst, Index axis);

template
void softmax_inplace<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &maxsumexp, Scalar alpha,
        const Tile<bf16_t> &dst, Index axis);

template
void softmax_inplace<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &maxsumexp, Scalar alpha,
        const Tile<fp16_t> &dst, Index axis);

} // namespace nntile::core
