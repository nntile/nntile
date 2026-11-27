/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/subtract_indexed_outputs.cc
 *
 * @version 1.1.0
 * */

#include "nntile/core/subtract_indexed_outputs.hh"
#include "nntile/nnhaul/ops/subtract_indexed_outputs.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

template<typename T>
void subtract_indexed_outputs_async(int worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<T> &dst, Index ignore_index)
{
    (void)worker_hint;
    if(labels.ndim != dst.ndim-1)
    {
        throw std::runtime_error("labels.ndim != dst.ndim-1");
    }
    for(Index i = 0; i < labels.ndim; ++i)
    {
        if(labels.shape[i] != dst.shape[i])
        {
            throw std::runtime_error("labels.shape[i] != dst.shape[i]");
        }
    }
    const Index n_labels = dst.shape[dst.ndim - 1];
    nnhaul::subtract_indexed_outputs.submit<std::tuple<T>>(
            n_labels, labels.nelems, ignore_index, val, labels, dst);
}

template<typename T>
void subtract_indexed_outputs(int worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<T> &dst, Index ignore_index)
{
    subtract_indexed_outputs_async<T>(worker_hint, val, labels, dst, ignore_index);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void subtract_indexed_outputs_async<fp32_t>(int worker_hint, Scalar val,
        const Tile<int64_t> &labels, const Tile<fp32_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs_async<fp32_fast_tf32_t>(int worker_hint, Scalar val,
        const Tile<int64_t> &labels, const Tile<fp32_fast_tf32_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs_async<fp32_fast_fp16_t>(int worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_fast_fp16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs_async<fp32_fast_bf16_t>(int worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_fast_bf16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs_async<fp64_t>(int worker_hint, Scalar val,
        const Tile<int64_t> &labels, const Tile<fp64_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs_async<bf16_t>(int worker_hint, Scalar val,
        const Tile<int64_t> &labels, const Tile<bf16_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs_async<fp16_t>(int worker_hint, Scalar val,
        const Tile<int64_t> &labels, const Tile<fp16_t> &dst,
        Index ignore_index);

// Explicit instantiation
template
void subtract_indexed_outputs<fp32_t>(int worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp32_fast_tf32_t>(int worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_fast_tf32_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp32_fast_fp16_t>(int worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_fast_fp16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp32_fast_bf16_t>(int worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_fast_bf16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp64_t>(int worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<fp64_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<bf16_t>(int worker_hint, Scalar val,
        const Tile<int64_t> &labels, const Tile<bf16_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs<fp16_t>(int worker_hint, Scalar val,
        const Tile<int64_t> &labels, const Tile<fp16_t> &dst,
        Index ignore_index);

} // namespace nntile::core
