/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/bias_outer.cc
 * Bias along outer axes operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-20
 * */

#include "nntile/tile/bias_outer.hh"
#include "nntile/starpu/bias_outer.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise bias_outer operation
template<typename T>
void bias_outer_async(T alpha, const Tile<T> &src, const Tile<T> &dst,
        Index axis)
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
    // Do nothing if alpha is zero
    if(alpha == 0.0)
    {
        return;
    }
    // Reshape inputs for simplicity: src -> (m,n), dst -> (m,k,n)
    Index m, n, k;
    m = dst.stride[axis];
    n = dst.matrix_shape[axis+1][1];
    k = dst.shape[axis];
    // Insert corresponding task
    starpu::bias_outer::submit<T>(m, n, k, alpha, src, dst);
}

//! Tile-wise bias_outer operation
template<typename T>
void bias_outer(T alpha, const Tile<T> &src, const Tile<T> &dst, Index axis)
{
    bias_outer_async<T>(alpha, src, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void bias_outer_async<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst, Index axis);

template
void bias_outer_async<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst, Index axis);

// Explicit instantiation of template
template
void bias_outer<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst, Index axis);

template
void bias_outer<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst, Index axis);

} // namespace tile
} // namespace nntile

