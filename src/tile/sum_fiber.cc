/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/sum_fiber.cc
 * Sums over slices into a fiber of a Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-26
 * */

#include "nntile/tile/sum_fiber.hh"
#include "nntile/starpu/sum_fiber.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise sum_fiber
template<typename T>
void sum_fiber_async(T alpha, const Tile<T> &src, T beta,
        const Tile<T> &dst, Index axis)
{
    // Check dimensions
    if(dst.ndim != 1)
    {
        throw std::runtime_error("dst.ndim != 1");
    }
    Index ndim = src.ndim;
    // Treat special case of ndim=0
    if(ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= ndim)
    {
        throw std::runtime_error("axis >= ndim");
    }
    // Check shapes
    if(dst.shape[0] != src.shape[axis])
    {
        throw std::runtime_error("dst.shape[0] != src.shape[axis]");
    }
    // Get sizes
    Index m, n, k;
    m = src.stride[axis];
    n = src.matrix_shape[axis+1][1];
    k = src.shape[axis];
    // Insert task
    starpu::sum_fiber::submit<T>(m, n, k, alpha, src, beta, dst);
}

//! Tile-wise sum_fiber
template<typename T>
void sum_fiber(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst,
        Index axis)
{
    sum_fiber_async<T>(alpha, src, beta, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void sum_fiber_async<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src,
        fp32_t beta, const Tile<fp32_t> &dst, Index axis);

template
void sum_fiber_async<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src,
        fp64_t beta, const Tile<fp64_t> &dst, Index axis);

// Explicit instantiation
template
void sum_fiber<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src, fp32_t beta,
        const Tile<fp32_t> &dst, Index axis);

template
void sum_fiber<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src, fp64_t beta,
        const Tile<fp64_t> &dst, Index axis);

} // namespace tile
} // namespace nntile

