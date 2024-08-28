/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/norm_slice.cc
 * Euclidean norms of fibers into a slice of a Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/norm_slice.hh"
#include "nntile/starpu/norm_slice.hh"

namespace nntile::tile
{

template<typename T>
void norm_slice_async(Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst,
        Index axis)
{
    // Check dimensions
    if(src.ndim-1 != dst.ndim)
    {
        throw std::runtime_error("src.ndim-1 != dst.ndim");
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
    for(Index i = 0; i < axis; i++)
    {
        if(src.shape[i] != dst.shape[i])
        {
            throw std::runtime_error("src.shape[i] != dst.shape[i]");
        }
    }
    for(Index i = axis+1; i < ndim; i++)
    {
        if(src.shape[i] != dst.shape[i-1])
        {
            throw std::runtime_error("src.shape[i] != dst.shape[i-1]");
        }
    }
    // Get sizes
    Index m, n, k;
    m = src.stride[axis];
    n = src.matrix_shape[axis+1][1];
    k = src.shape[axis];
    // Insert task
    starpu::norm_slice::submit<T>(m, n, k, alpha, src, beta, dst);
}

template<typename T>
void norm_slice(Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst,
        Index axis)
{
    norm_slice_async<T>(alpha, src, beta, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void norm_slice_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src,
        Scalar beta, const Tile<fp32_t> &dst, Index axis);

template
void norm_slice_async<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        Scalar beta, const Tile<fp32_fast_tf32_t> &dst, Index axis);

template
void norm_slice_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src,
        Scalar beta, const Tile<fp64_t> &dst, Index axis);

template
void norm_slice_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst, Index axis);

// Explicit instantiation
template
void norm_slice<fp32_t>(Scalar alpha, const Tile<fp32_t> &src, Scalar beta,
        const Tile<fp32_t> &dst, Index axis);

template
void norm_slice<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst, Index axis);

template
void norm_slice<fp64_t>(Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst, Index axis);

template
void norm_slice<bf16_t>(Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst, Index axis);

} // namespace nntile::tile
