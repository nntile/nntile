/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/scale_inplace.cc
 * Inplace scale of Tile<T>
 *
 * @version 2.0.0
 * */

#include "nntile/core/scale_inplace.hh"
#include "nntile/nnhaul/ops/scale_inplace.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

//! Tile-wise scale_inplace operation
template<typename T>
void scale_inplace_async(int worker_hint, Scalar alpha, const Tile<T> &data)
{
    // Insert task
    nnhaul::scale_inplace.submit<std::tuple<T>>(worker_hint, data.nelems, alpha, data);
}

//! Tile-wise scale_inplace operation
template<typename T>
void scale_inplace(int worker_hint, Scalar alpha, const Tile<T> &data)
{
    scale_inplace_async<T>(worker_hint, alpha, data);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void scale_inplace_async<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &data);

template
void scale_inplace_async<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &data);

template
void scale_inplace_async<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &data);

template
void scale_inplace_async<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &data);

template
void scale_inplace_async<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &data);

template
void scale_inplace_async<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &data);

template
void scale_inplace_async<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &data);

// Explicit instantiation
template
void scale_inplace<fp32_t>(int worker_hint, Scalar alpha, const Tile<fp32_t> &data);

template
void scale_inplace<fp32_fast_tf32_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &data);

template
void scale_inplace<fp32_fast_fp16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &data);

template
void scale_inplace<fp32_fast_bf16_t>(int worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &data);

template
void scale_inplace<fp64_t>(int worker_hint, Scalar alpha, const Tile<fp64_t> &data);

template
void scale_inplace<fp16_t>(int worker_hint, Scalar alpha, const Tile<fp16_t> &data);

template
void scale_inplace<bf16_t>(int worker_hint, Scalar alpha, const Tile<bf16_t> &data);

} // namespace nntile::core
