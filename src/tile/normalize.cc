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
 * @date 2022-08-31
 * */

#include "nntile/tile/normalize.hh"
#include "nntile/starpu/normalize.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise average and deviation from sum and scaled sum of squares
template<typename T>
void normalize_async(const StarpuVariableHandle &gamma_beta,
        const Tile<T> &sumnorm, const Tile<T> &dst, Index l, T eps,
        Index axis)
{
    // Check inputs
    if(sumnorm.ndim != dst.ndim)
    {
        throw std::runtime_error("sumnorm.ndim != dst.ndim");
    }
    // Input shape dimension shall be at least 1
    if(sumnorm.ndim == 0)
    {
        throw std::runtime_error("sumnorm.ndim == 0");
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
    // Check shapes
    if(sumnorm.shape[0] != 2)
    {
        throw std::runtime_error("sumnorm.shape[0] != 2");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != sumnorm.shape[i+1])
        {
            throw std::runtime_error("dst.shape[i] != sumnorm.shape[i+1]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != sumnorm.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != sumnorm.shape[i]");
        }
    }
    // Reshape inputs for simplicity: sumnorm -> (2,m,n), dst -> (m,k,n)
    // dst is a part of (m,l,n) tensor
    Index m, n, k;
    if(axis == 0)
    {
        m = 1;
        n = sumnorm.nelems / 2; // 2 elements per single n
        k = dst.shape[0];
    }
    else if(axis == dst.ndim-1)
    {
        m = sumnorm.nelems / 2; // 2 elements per single m
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
    starpu::normalize::submit<T>(m, n, k, l, eps, gamma_beta, sumnorm, dst);
}

//! Tile-wise average and deviation from sum and scaled sum of squares
template<typename T>
void normalize(const StarpuVariableHandle &gamma_beta,
        const Tile<T> &sumnorm, const Tile<T> &dst, Index l, T eps,
        Index axis)
{
    normalize_async<T>(gamma_beta, sumnorm, dst, l, eps, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void normalize(const StarpuVariableHandle &gamma_beta,
        const Tile<fp32_t> &sumnorm, const Tile<fp32_t> &dst, Index l,
        fp32_t eps, Index axis);

template
void normalize(const StarpuVariableHandle &gamma_beta,
        const Tile<fp64_t> &sumnorm, const Tile<fp64_t> &dst, Index l,
        fp64_t eps, Index axis);

} // namespace tile
} // namespace nntile

