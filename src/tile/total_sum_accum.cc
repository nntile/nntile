/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/total_sum_accum.cc
 * Total sum accumulating for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/total_sum_accum.hh"
#include "nntile/starpu/total_sum_accum.hh"

namespace nntile::tile
{

template<typename T>
void total_sum_accum_async(Scalar alpha, const Tile<T> &logsumexp,
        const Tile<T> &src, const Tile<int64_t> &labels,
        const Tile<fp32_t> &val)
// TODO - add description
{
    // Check dimensions
    if(logsumexp.ndim != labels.ndim)
    {
        throw std::runtime_error("logsumexp.ndim != labels.ndim");
    }
    if(logsumexp.ndim != src.ndim-1)
    {
        throw std::runtime_error("logsumexp.ndim != src.ndim-1");
    }
    if(val.ndim != 0)
    {
        throw std::runtime_error("val.ndim != 0");
    }
    // Check shapes
    for(Index i = 0; i < labels.ndim; ++i)
    {
        if(logsumexp.shape[i] != labels.shape[i])
        {
            throw std::runtime_error("logsumexp.shape[i] != labels.shape[i]");
        }
        if(labels.shape[i] != src.shape[i+1])
        {
            throw std::runtime_error("labels.shape[i] != src.shape[i+1]");
        }
    }
    // Insert task
    starpu::total_sum_accum::submit<T>(alpha, src.shape[0], logsumexp.nelems,
            logsumexp, src, labels, val);
}

//! Tile-wise max and sum of exponents along single given axis
template<typename T>
void total_sum_accum(Scalar alpha, const Tile<T> &logsumexp,
        const Tile<T> &src, const Tile<int64_t> &class_labels,
        const Tile<fp32_t> &val)
{
    total_sum_accum_async<T>(alpha, logsumexp, src, class_labels, val);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void total_sum_accum_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &logsumexp,
        const Tile<fp32_t> &src, const Tile<int64_t> &class_labels,
        const Tile<fp32_t> &val);

template
void total_sum_accum_async<fp32_fast_tf32_t>(Scalar alpha,
        const Tile<fp32_fast_tf32_t> &logsumexp,
        const Tile<fp32_fast_tf32_t> &src, const Tile<int64_t> &class_labels,
        const Tile<fp32_t> &val);

template
void total_sum_accum_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &logsumexp,
        const Tile<fp64_t> &src, const Tile<int64_t> &class_labels,
        const Tile<fp32_t> &val);

template
void total_sum_accum_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &logsumexp,
        const Tile<bf16_t> &src, const Tile<int64_t> &class_labels,
        const Tile<fp32_t> &val);

// Explicit instantiation
template
void total_sum_accum<fp32_t>(Scalar alpha, const Tile<fp32_t> &logsumexp,
        const Tile<fp32_t> &src, const Tile<int64_t> &class_labels,
        const Tile<fp32_t> &val);

template
void total_sum_accum<fp32_fast_tf32_t>(Scalar alpha,
        const Tile<fp32_fast_tf32_t> &logsumexp,
        const Tile<fp32_fast_tf32_t> &src, const Tile<int64_t> &class_labels,
        const Tile<fp32_t> &val);

template
void total_sum_accum<fp64_t>(Scalar alpha, const Tile<fp64_t> &logsumexp,
        const Tile<fp64_t> &src, const Tile<int64_t> &class_labels,
        const Tile<fp32_t> &val);

template
void total_sum_accum<bf16_t>(Scalar alpha, const Tile<bf16_t> &logsumexp,
        const Tile<bf16_t> &src, const Tile<int64_t> &class_labels,
        const Tile<fp32_t> &val);

} // namespace nntile::tile
