/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/normalize.cc
 * Normalize operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/normalize.hh"
#include "nntile/starpu/normalize.hh"

namespace nntile::tile
{

//! Tile-wise average and deviation from sum and scaled sum of squares
template<typename T>
void normalize_async(const Tile<T> &gamma_beta, const Tile<T> &sumnorm,
        const Tile<T> &dst, Index size, Scalar eps, Index axis)
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
    if(size <= 0)
    {
        throw std::runtime_error("size <= 0");
    }
    // Check regularization
    if(eps <= 0)
    {
        throw std::runtime_error("eps <= 0");
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
    // dst is a part of (m,size,n) tensor
    Index m, n, k;
    m = dst.stride[axis];
    n = dst.matrix_shape[axis+1][1];
    k = dst.shape[axis];
    // Insert task
    starpu::normalize::submit<T>(m, n, k, size, eps, gamma_beta, sumnorm, dst);
}

//! Tile-wise average and deviation from sum and scaled sum of squares
template<typename T>
void normalize(const Tile<T> &gamma_beta, const Tile<T> &sumnorm,
        const Tile<T> &dst, Index size, Scalar eps, Index axis)
{
    normalize_async<T>(gamma_beta, sumnorm, dst, size, eps, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void normalize_async<fp32_t>(const Tile<fp32_t> &gamma_beta,
        const Tile<fp32_t> &sumnorm, const Tile<fp32_t> &dst, Index size,
        Scalar eps, Index axis);

template
void normalize_async<fp64_t>(const Tile<fp64_t> &gamma_beta,
        const Tile<fp64_t> &sumnorm, const Tile<fp64_t> &dst, Index size,
        Scalar eps, Index axis);

// Explicit instantiation
template
void normalize<fp32_t>(const Tile<fp32_t> &gamma_beta,
        const Tile<fp32_t> &sumnorm, const Tile<fp32_t> &dst, Index size,
        Scalar eps, Index axis);

template
void normalize<fp64_t>(const Tile<fp64_t> &gamma_beta,
        const Tile<fp64_t> &sumnorm, const Tile<fp64_t> &dst, Index size,
        Scalar eps, Index axis);

} // namespace nntile::tile
