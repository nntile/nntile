/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/add_fiber.cc
 * Bias operation over fibers from a slice of a Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-26
 * */

#include "nntile/tile/add_fiber.hh"
#include "nntile/starpu/add_fiber.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise add_fiber operation
template<typename T>
void add_fiber_async(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst,
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
    starpu::add_fiber::submit<T>(m, n, k, alpha, src, beta, dst);
}

//! Tile-wise add_fiber operation
template<typename T>
void add_fiber(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst,
        Index axis)
{
    add_fiber_async<T>(alpha, src, beta, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void add_fiber_async<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src,
        fp32_t beta, const Tile<fp32_t> &dst, Index axis);

template
void add_fiber_async<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src,
        fp64_t beta, const Tile<fp64_t> &dst, Index axis);

// Explicit instantiation of template
template
void add_fiber<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src, fp32_t beta,
        const Tile<fp32_t> &dst, Index axis);

template
void add_fiber<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src, fp64_t beta,
        const Tile<fp64_t> &dst, Index axis);

} // namespace tile
} // namespace nntile

