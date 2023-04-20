/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/norm.cc
 * Norm of Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-20
 * */

#include "nntile/tile/norm.hh"
#include "nntile/starpu/norm.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise norm single given axis
template<typename T>
void norm_async(T alpha, const Tile<T> &src, T beta, const Tile<T> &norm_dst,
        Index axis)
{
    // Check dimensions
    if(src.ndim-1 != norm_dst.ndim)
    {
        throw std::runtime_error("src.ndim-1 != norm_dst.ndim");
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
    // Check shapes of src and norm_dst
    for(Index i = 0; i < axis; i++)
    {
        if(src.shape[i] != norm_dst.shape[i])
        {
            throw std::runtime_error("src.shape[i] != norm_dst.shape[i]");
        }
    }
    for(Index i = axis+1; i < ndim; i++)
    {
        if(src.shape[i] != norm_dst.shape[i-1])
        {
            throw std::runtime_error("src.shape[i] != norm_dst.shape[i-1]");
        }
    }
    // Get sizes
    Index m, n, k;
    m = src.stride[axis];
    n = src.matrix_shape[axis+1][1];
    k = src.shape[axis];
    // Insert task
    starpu::norm::submit<T>(m, n, k, alpha, src, beta, norm_dst);
}

//! Tile-wise norm along single given axis
template<typename T>
void norm(T alpha, const Tile<T> &src, T beta, const Tile<T> &norm_dst,
        Index axis)
{
    norm_async<T>(alpha, src, beta, norm_dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void norm_async<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src, fp32_t beta,
        const Tile<fp32_t> &norm_dst, Index axis);

template
void norm_async<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src, fp64_t beta,
        const Tile<fp64_t> &norm_dst, Index axis);

// Explicit instantiation
template
void norm<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src, fp32_t beta,
        const Tile<fp32_t> &norm_dst, Index axis);

template
void norm<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src, fp64_t beta,
        const Tile<fp64_t> &norm_dst, Index axis);

} // namespace tile
} // namespace nntile

