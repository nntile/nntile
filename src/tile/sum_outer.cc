/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/sum_outer.cc
 * Sum of slices of a Tile<T> (outer version)
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-24
 * */

#include "nntile/tile/sum_outer.hh"
#include "nntile/starpu/sum_outer.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise sum over all axes except given axis
template<typename T>
void sum_outer_async(T alpha, const Tile<T> &src, T beta,
        const Tile<T> &sum_dst, Index axis)
{
    // Check dimensions
    if(sum_dst.ndim != 1)
    {
        throw std::runtime_error("sum_dst.ndim != 1");
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
    if(sum_dst.shape[0] != src.shape[axis])
    {
        throw std::runtime_error("sum_dst.shape[0] != src.shape[axis]");
    }
    // Get sizes
    Index m, n, k;
    m = src.stride[axis];
    n = src.matrix_shape[axis+1][1];
    k = src.shape[axis];
    // Insert task
    starpu::sum_outer::submit<T>(m, n, k, alpha, src, beta, sum_dst);
}

//! Tile-wise sum along all axes axcept single given axis
template<typename T>
void sum_outer(T alpha, const Tile<T> &src, T beta, const Tile<T> &sum_dst,
        Index axis)
{
    sum_outer_async<T>(alpha, src, beta, sum_dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void sum_outer_async<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src,
        fp32_t beta, const Tile<fp32_t> &sum_dst, Index axis);

template
void sum_outer_async<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src,
        fp64_t beta, const Tile<fp64_t> &sum_dst, Index axis);

// Explicit instantiation
template
void sum_outer<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src, fp32_t beta,
        const Tile<fp32_t> &sum_dst, Index axis);

template
void sum_outer<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src, fp64_t beta,
        const Tile<fp64_t> &sum_dst, Index axis);

} // namespace tile
} // namespace nntile

