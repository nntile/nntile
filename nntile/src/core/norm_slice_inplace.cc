/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/norm_slice_inplace.cc
 * Euclidean norms of fibers into a slice of a Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/norm_slice_inplace.hh"
#include "nntile/nnhaul/ops/norm_slice_inplace.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

template<typename T>
void norm_slice_inplace_async(int worker_hint, Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst,
        Index axis, int redux)
{
    if(redux != 0)
    {
        throw std::runtime_error("redux != 0 is not supported");
    }
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
    m = src.matrix_shape[axis+1][1];
    n = src.matrix_shape[axis][0];
    k = src.shape[axis];
    // Insert task
    nnhaul::norm_slice_inplace.submit<std::tuple<T>>(worker_hint, m, n, k, alpha, src,
            beta, dst, 0);
  // redux ignored for now
}

template<typename T>
void norm_slice_inplace(int worker_hint, Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst,
        Index axis, int redux)
{
    if(redux != 0)
    {
        throw std::runtime_error("redux != 0 is not supported");
    }
    norm_slice_inplace_async<T>(worker_hint, alpha, src, beta, dst, axis, redux);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void norm_slice_inplace_async<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        Scalar beta, const Tile<fp32_t> &dst, Index axis, int redux);

template
void norm_slice_inplace_async<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        Scalar beta, const Tile<fp32_fast_tf32_t> &dst, Index axis, int redux);

template
void norm_slice_inplace_async<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src, Scalar beta,
        const Tile<fp32_fast_fp16_t> &dst, Index axis, int redux);

template
void norm_slice_inplace_async<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src, Scalar beta,
        const Tile<fp32_fast_bf16_t> &dst, Index axis, int redux);

template
void norm_slice_inplace_async<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        Scalar beta, const Tile<fp64_t> &dst, Index axis, int redux);

template
void norm_slice_inplace_async<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst, Index axis, int redux);

template
void norm_slice_inplace_async<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &src, Scalar beta,
        const Tile<fp16_t> &dst, Index axis, int redux);

// Explicit instantiation
template
void norm_slice_inplace<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &src, Scalar beta,
        const Tile<fp32_t> &dst, Index axis, int redux);

template
void norm_slice_inplace<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst, Index axis, int redux);

template
void norm_slice_inplace<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src, Scalar beta,
        const Tile<fp32_fast_fp16_t> &dst, Index axis, int redux);

template
void norm_slice_inplace<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src, Scalar beta,
        const Tile<fp32_fast_bf16_t> &dst, Index axis, int redux);

template
void norm_slice_inplace<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst, Index axis, int redux);

template
void norm_slice_inplace<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst, Index axis, int redux);

template
void norm_slice_inplace<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &src, Scalar beta,
        const Tile<fp16_t> &dst, Index axis, int redux);

} // namespace nntile::core
