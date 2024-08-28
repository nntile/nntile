/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/maxsumexp.cc
 * Max and sum of exponents of Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/maxsumexp.hh"
#include "nntile/starpu/maxsumexp.hh"

namespace nntile::tile
{

//! Tile-wise max and sum of exponents along single given axis
template<typename T>
void maxsumexp_async(const Tile<T> &src, const Tile<T> &dst, Index axis)
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
    m = src.stride[axis];
    n = src.matrix_shape[axis+1][1];
    k = src.shape[axis];
    // Insert task
    starpu::maxsumexp::submit<T>(m, n, k, src, dst);
}

//! Tile-wise max and sum of exponents along single given axis
template<typename T>
void maxsumexp(const Tile<T> &src, const Tile<T> &dst, Index axis)
{
    maxsumexp_async<T>(src, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void maxsumexp_async<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst,
        Index axis);

template
void maxsumexp_async<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &src, const Tile<fp32_fast_tf32_t> &dst,
        Index axis);

template
void maxsumexp_async<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst,
        Index axis);

template
void maxsumexp_async<bf16_t>(const Tile<bf16_t> &src, const Tile<bf16_t> &dst,
        Index axis);

// Explicit instantiation
template
void maxsumexp<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst,
        Index axis);

template
void maxsumexp<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &src, const Tile<fp32_fast_tf32_t> &dst,
        Index axis);

template
void maxsumexp<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst,
        Index axis);

template
void maxsumexp<bf16_t>(const Tile<bf16_t> &src, const Tile<bf16_t> &dst,
        Index axis);

} // namespace nntile::tile
