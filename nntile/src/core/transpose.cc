/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/transpose.cc
 * Transpose operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/transpose.hh"
#include "nntile/nnhaul/ops/transpose.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

//! Tile-wise transpose operation
template<typename T>
void transpose_async(int worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst,
        Index ndim)
{
    // Check dimensions
    if(ndim <= 0 or ndim >= src.ndim)
    {
        throw std::runtime_error("ndim <= 0 or ndim >= src.ndim");
    }
    if(dst.ndim != src.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }
    // Check shapes of tiles
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(src.shape[(i+ndim) % dst.ndim] != dst.shape[i])
        {
            throw std::runtime_error("src.shape[(i+ndim) % dst.ndim] != "
                    "dst.shape[i]");
        }
    }
    nnhaul::transpose.submit<std::tuple<T>>(worker_hint,
            src.matrix_shape[ndim][1],
            src.matrix_shape[ndim][0], alpha, src, dst);
}

//! Tile-wise transpose operation
template<typename T>
void transpose(int worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst,
        Index ndim)
{
    transpose_async<T>(worker_hint, alpha, src, dst, ndim);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation of template
template
void transpose_async<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst, Index ndim);

template
void transpose_async<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst, Index ndim);

template
void transpose_async<fp32_fast_tf32_t>(int worker_hint, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst, Index ndim);

template
void transpose_async<fp32_fast_fp16_t>(int worker_hint, Scalar alpha,
        const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst, Index ndim);

template
void transpose_async<fp32_fast_bf16_t>(int worker_hint, Scalar alpha,
        const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst, Index ndim);

template
void transpose_async<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst, Index ndim);

template
void transpose_async<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst, Index ndim);

// Explicit instantiation of template
template
void transpose<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst, Index ndim);

template
void transpose<fp32_fast_tf32_t>(int worker_hint, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst, Index ndim);

template
void transpose<fp32_fast_fp16_t>(int worker_hint, Scalar alpha,
        const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst, Index ndim);

template
void transpose<fp32_fast_bf16_t>(int worker_hint, Scalar alpha,
        const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst, Index ndim);

template
void transpose<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst, Index ndim);

template
void transpose<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst, Index ndim);

template
void transpose<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst, Index ndim);

} // namespace nntile::core
