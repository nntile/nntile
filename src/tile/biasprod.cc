/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/biasprod.cc
 * Bias-like product operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-19
 * */

#include "nntile/tile/biasprod.hh"
#include "nntile/starpu/biasprod.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise biasprod operation
template<typename T>
void biasprod_async(const Tile<T> &src, const Tile<T> &dst, Index axis)
{
    // Check dimensions
    if(dst.ndim != src.ndim+1)
    {
        throw std::runtime_error("dst.ndim != src.ndim+1");
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
    // Check shapes of tiles
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i-1])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i-1]");
        }
    }
    // Reshape inputs for simplicity: src -> (m,n), dst -> (m,k,n)
    Index m, n, k;
    if(axis == 0)
    {
        m = 1;
        n = src.nelems;
        k = dst.shape[0];
    }
    else if(axis == dst.ndim-1)
    {
        m = src.nelems;
        n = 1;
        k = dst.shape[axis];
    }
    else
    {
        m = dst.stride[axis];
        n = dst.matrix_shape[axis+1][1];
        k = dst.shape[axis];
    }
    // Insert corresponding task
    starpu::biasprod::submit<T>(m, n, k, src, dst);
}

//! Tile-wise biasprod operation
template<typename T>
void biasprod(const Tile<T> &src, const Tile<T> &dst, Index axis)
{
    biasprod_async<T>(src, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void biasprod_async<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst,
        Index axis);

template
void biasprod_async<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst,
        Index axis);

// Explicit instantiation of template
template
void biasprod<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst,
        Index axis);

template
void biasprod<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst,
        Index axis);

} // namespace tile
} // namespace nntile

