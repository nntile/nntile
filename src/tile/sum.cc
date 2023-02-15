/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/sum.cc
 * Sum and Euclidian norm of Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev and K. Sozykin
 * @date 2022-12-02
 * */

#include "nntile/tile/sum.hh"
#include "nntile/starpu/sum.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise sum  single given axis
template<typename T>
void sum_async(const Tile<T> &src, const Tile<T> &dst, Index axis)
{
    // Check dimensions
    if(src.ndim != dst.ndim)
    {
        throw std::runtime_error("src.ndim != dst.ndim");
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
    // Check shapes of src and dst
    if(dst.shape[0] != 2)
    {
        throw std::runtime_error("dst.shape[0] != 2");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(src.shape[i] != dst.shape[i+1])
        {
            throw std::runtime_error("src.shape[i] != dst.shape[i+1]");
        }
    }
    for(Index i = axis+1; i < src.ndim; ++i)
    {
        if(src.shape[i] != dst.shape[i])
        {
            throw std::runtime_error("src.shape[i] != dst.shape[i]");
        }
    }
    // Get sizes
    Index m, n, k;
    if(axis == 0)
    {
        m = 1;
        n = dst.nelems / 2;
        k = src.shape[0];
    }
    else if(axis == ndim-1)
    {
        m = dst.nelems / 2;
        n = 1;
        k = src.shape[axis];
    }
    else
    {
        m = src.stride[axis];
        n = src.matrix_shape[axis+1][1];
        k = src.shape[axis];
    }
    // Insert task
    starpu::sum::submit<T>(m, n, k, src, dst);
}

//! Tile-wise sum and scaled sum of squares along single given axis
template<typename T>
void sum(const Tile<T> &src, const Tile<T> &dst, Index axis)
{
    sum_async<T>(src, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void sum_async<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst,
        Index axis);

template
void sum_async<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst,
        Index axis);

// Explicit instantiation
template
void sum<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst,
        Index axis);

template
void sum<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst,
        Index axis);

} // namespace tile
} // namespace nntile

