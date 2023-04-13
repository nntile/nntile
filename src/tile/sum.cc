/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/sum.cc
 * Sum of Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Konstantin Sozykin
 * @date 2023-04-13
 * */

#include "nntile/tile/sum.hh"
#include "nntile/starpu/sum.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise sum  single given axis
template<typename T>
void sum_async(T alpha, const Tile<T> &src, T beta, const Tile<T> &sum_dst,
        Index axis)
{
    // Check dimensions
    if(src.ndim - 1 != sum_dst.ndim) // before was src.ndim != sum_dst.ndim
    {
        throw std::runtime_error("src.ndim -1 != sum_dst.ndim");
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

    // check if axis consisted, using two pointers
    for(Index i = 0, j = 0; i < src.ndim; i++)
    {
        if (i == axis) {
            continue;
        }
        if(src.shape[i] != sum_dst.shape[j])
        {
            throw std::runtime_error("src.shape[i] != sum_dst.shape[j]");
        }
        j++;
    }
    // Get sizes
    Index m, n, k;
    if(axis == 0)
    {
        m = 1;
        n = sum_dst.nelems; 
        k = src.shape[0];
    }
    else if(axis == ndim-1)
    {
        m = sum_dst.nelems;
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
    starpu::sum::submit<T>(m, n, k, alpha, src, beta, sum_dst);
}

//! Tile-wise sum and scaled sum of squares along single given axis
template<typename T>
void sum(T alpha, const Tile<T> &src, T beta, const Tile<T> &sum_dst,
        Index axis)
{
    sum_async<T>(alpha, src, beta, sum_dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void sum_async<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src, fp32_t beta,
        const Tile<fp32_t> &sum_dst, Index axis);

template
void sum_async<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src, fp64_t beta,
        const Tile<fp64_t> &sum_dst, Index axis);

// Explicit instantiation
template
void sum<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src, fp32_t beta,
        const Tile<fp32_t> &sum_dst, Index axis);

template
void sum<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src, fp64_t beta,
        const Tile<fp64_t> &sum_dst, Index axis);

} // namespace tile
} // namespace nntile

