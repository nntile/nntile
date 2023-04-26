/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/add_slice.cc
 * Bias operation over fibers from a slice of a Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-26
 * */

#include "nntile/tile/add_slice.hh"
#include "nntile/starpu/add_slice.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise add_slice operation
template<typename T>
void add_slice_async(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst,
        Index axis)
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
    m = dst.stride[axis];
    n = dst.matrix_shape[axis+1][1];
    k = dst.shape[axis];
    // Insert corresponding task
    starpu::add_slice::submit<T>(m, n, k, alpha, src, beta, dst);
}

//! Tile-wise add_slice operation
template<typename T>
void add_slice(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst,
        Index axis)
{
    add_slice_async<T>(alpha, src, beta, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void add_slice_async<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src,
        fp32_t beta, const Tile<fp32_t> &dst, Index axis);

template
void add_slice_async<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src,
        fp64_t beta, const Tile<fp64_t> &dst, Index axis);

// Explicit instantiation of template
template
void add_slice<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src, fp32_t beta,
        const Tile<fp32_t> &dst, Index axis);

template
void add_slice<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src, fp64_t beta,
        const Tile<fp64_t> &dst, Index axis);

} // namespace tile
} // namespace nntile

