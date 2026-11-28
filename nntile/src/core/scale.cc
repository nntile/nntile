/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/scale.cc
 * Scale operation for Tile<T>
 *
 * @version 2.0.0
 * */

#include "nntile/core/scale.hh"
#include "nntile/nnhaul/ops/scale.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

//! Tile-wise scale operation
template<typename T>
void scale_async(int worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst)
{
    // Check dimensions
    if(dst.ndim != src.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }
    // Check shapes of tiles
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
    }
    // Insert corresponding task
    nnhaul::scale.submit<std::tuple<T>>(worker_hint, src.nelems, alpha, src, dst);
}

//! Tile-wise scale operation
template<typename T>
void scale(int worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst)
{
    scale_async<T>(worker_hint, alpha, src, dst);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation of template
template
void scale_async<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void scale_async<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void scale_async<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void scale_async<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void scale_async<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

template
void scale_async<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst);

template
void scale_async<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst);

// Explicit instantiation of template
template
void scale<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void scale<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void scale<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void scale<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void scale<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

template
void scale<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst);

template
void scale<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst);

} // namespace nntile::core
