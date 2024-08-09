/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/softmax_inplace.cc
 * softmax_inplace operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/softmax_inplace.hh"
#include "nntile/starpu/softmax_inplace.hh"

namespace nntile::tile
{

template<typename T>
void softmax_inplace_async(const Tile<T> &maxsumexp, Scalar alpha,
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
    // Check shapes
    if(maxsumexp.shape[0] != 2)
    {
        throw std::runtime_error("maxsumexp.shape[0] != 2");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != maxsumexp.shape[i+1])
        {
            throw std::runtime_error("dst.shape[i] != maxsumexp.shape[i+1]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != maxsumexp.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != maxsumexp.shape[i]");
        }
    }
    // Reshape inputs for simplicity: maxsumexp -> (2,m,n), dst -> (m,k,n)
    // dst is a part of (m,l,n) tensor
    Index m, n, k;
    m = dst.stride[axis];
    n = dst.matrix_shape[axis+1][1];
    k = dst.shape[axis];
    // Insert task
    starpu::softmax_inplace::submit<T>(m, n, k, maxsumexp, alpha, dst);
}

//! Tile-wise average and deviation from sum and scaled sum of squares
template<typename T>
void softmax_inplace(const Tile<T> &maxsumexp, Scalar alpha, const Tile<T> &dst,
        Index axis)
{
    softmax_inplace_async<T>(maxsumexp, alpha, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void softmax_inplace_async<fp32_t>(const Tile<fp32_t> &maxsumexp, Scalar alpha,
        const Tile<fp32_t> &dst, Index axis);

template
void softmax_inplace_async<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &maxsumexp, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &dst, Index axis);

template
void softmax_inplace_async<fp64_t>(const Tile<fp64_t> &maxsumexp, Scalar alpha,
        const Tile<fp64_t> &dst, Index axis);

template
void softmax_inplace_async<bf16_t>(const Tile<bf16_t> &maxsumexp, Scalar alpha,
        const Tile<bf16_t> &dst, Index axis);

// Explicit instantiation
template
void softmax_inplace<fp32_t>(const Tile<fp32_t> &maxsumexp, Scalar alpha,
        const Tile<fp32_t> &dst, Index axis);

template
void softmax_inplace<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &maxsumexp, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &dst, Index axis);

template
void softmax_inplace<fp64_t>(const Tile<fp64_t> &maxsumexp, Scalar alpha,
        const Tile<fp64_t> &dst, Index axis);

template
void softmax_inplace<bf16_t>(const Tile<bf16_t> &maxsumexp, Scalar alpha,
        const Tile<bf16_t> &dst, Index axis);

} // namespace nntile::tile
