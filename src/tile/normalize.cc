/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/normalize.cc
 * Normalize operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-26
 * */

#include "nntile/tile/normalize.hh"
#include "nntile/starpu/normalize.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise average and deviation from sum and scaled sum of squares
template<typename T>
void normalize_async(const Tile<T> &gamma_beta, const Tile<T> &src,
        const Tile<T> &dst, Index l, T eps, Index axis)
{
    // Check gamma_beta
    if(gamma_beta.shape.size() != 1)
    {
        throw std::runtime_error("gamma_beta.shape.size() != 1");
    }
    if(gamma_beta.shape[0] != 2)
    {
        throw std::runtime_error("gamma_beta.shape[0] != 2");
    }
    // Check dimensions
    if(src.ndim != dst.ndim)
    {
        throw std::runtime_error("src.ndim != dst.ndim");
    }
    // Input shape dimension shall be at least 1
    if(src.ndim == 0)
    {
        throw std::runtime_error("src.ndim == 0");
    }
    // Check number of elements
    if(l <= 0)
    {
        throw std::runtime_error("l <= 0");
    }
    // Check regularization
    if(eps < 0)
    {
        throw std::runtime_error("eps < 0");
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
    if(src.shape[0] != 2)
    {
        throw std::runtime_error("src.shape[0] != 2");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != src.shape[i+1])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i+1]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
    }
    // Reshape inputs for simplicity: src -> (2,m,n), dst -> (m,k,n)
    // dst is a part of (m,l,n) tensor
    Index m, n, k;
    if(axis == 0)
    {
        m = 1;
        n = src.nelems / 2; // 2 elements per single n
        k = dst.shape[0];
    }
    else if(axis == dst.ndim-1)
    {
        m = src.nelems / 2; // 2 elements per single m
        n = 1;
        k = dst.shape[axis];
    }
    else
    {
        m = dst.stride[axis];
        n = dst.matrix_shape[axis+1][1];
        k = dst.shape[axis];
    }
    // Insert task
    starpu::normalize::submit<T>(m, n, k, l, eps, gamma_beta, src, dst);
}

//! Tile-wise average and deviation from sum and scaled sum of squares
template<typename T>
void normalize(const Tile<T> &gamma_beta, const Tile<T> &src,
        const Tile<T> &dst, Index l, T eps, Index axis)
{
    normalize_async<T>(gamma_beta, src, dst, l, eps, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void normalize<fp32_t>(const Tile<fp32_t> &gamma_beta, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst, Index l, fp32_t eps, Index axis);

template
void normalize<fp64_t>(const Tile<fp64_t> &gamma_beta, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst, Index l, fp64_t eps, Index axis);

} // namespace tile
} // namespace nntile

