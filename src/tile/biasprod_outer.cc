/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/biasprod_outer.cc
 * Bias-like product along outer axes operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-20
 * */

#include "nntile/tile/biasprod_outer.hh"
#include "nntile/starpu/biasprod_outer.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise biasprod_outer operation
template<typename T>
void biasprod_outer_async(const Tile<T> &src, const Tile<T> &dst, Index axis)
{
    // Check dimensions
    if(src.ndim != 1)
    {
        throw std::runtime_error("src.ndim != 1");
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
    if(src.shape[0] != dst.shape[axis])
    {
        throw std::runtime_error("src.shape[0] != dst.shape[axis]");
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
    starpu::biasprod_outer::submit<T>(m, n, k, src, dst);
}

//! Tile-wise biasprod_outer operation
template<typename T>
void biasprod_outer(const Tile<T> &src, const Tile<T> &dst, Index axis)
{
    biasprod_outer_async<T>(src, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void biasprod_outer_async<fp32_t>(const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst, Index axis);

template
void biasprod_outer_async<fp64_t>(const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst, Index axis);

// Explicit instantiation of template
template
void biasprod_outer<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst,
        Index axis);

template
void biasprod_outer<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst,
        Index axis);

} // namespace tile
} // namespace nntile

