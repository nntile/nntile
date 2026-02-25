/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/total_sum_accum.cc
 * Total sum accumulating of Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/total_sum_accum.hh"
#include "nntile/tile/total_sum_accum.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Compute total_sum_accum
template<typename T>
void total_sum_accum_async(Scalar alpha, const Tensor<T> &logsumexp,
        const Tensor<T> &src, const Tensor<int64_t> &labels,
        const Tensor<fp32_t> &val, Index ignore_index)
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
        if(logsumexp.basetile_shape[i] != labels.basetile_shape[i])
        {
            throw std::runtime_error("logsumexp.basetile_shape[i] != "
                    "labels.basetile_shape[i]");
        }
        if(labels.shape[i] != src.shape[i+1])
        {
            throw std::runtime_error("labels.shape[i] != src.shape[i+1]");
        }
        if(labels.basetile_shape[i] != src.basetile_shape[i+1])
        {
            throw std::runtime_error("labels.basetile_shape[i] != "
                    "src.basetile_shape[i+1]");
        }
    }
    if(src.basetile_shape[0] != src.shape[0])
    {
        throw std::runtime_error("src.basetile_shape[0] != src.shape[0]");
    }
    // Do actual calculations
    auto val_tile_handle = val.get_tile_handle(0);
    auto val_tile = val.get_tile(0);
    for(Index i = 0; i < labels.grid.nelems; ++i)
    {
        auto logsumexp_tile = logsumexp.get_tile(i);
        auto src_tile = src.get_tile(i);
        auto labels_tile = labels.get_tile(i);
        tile::total_sum_accum_async<T>(alpha, logsumexp_tile, src_tile,
                labels_tile, val_tile, ignore_index);
    }
    val_tile_handle.mpi_flush();
}

template<typename T>
void total_sum_accum(Scalar alpha, const Tensor<T> &logsumexp,
        const Tensor<T> &src, const Tensor<int64_t> &labels,
        const Tensor<fp32_t> &val, Index ignore_index)
{
    total_sum_accum_async<T>(alpha, logsumexp, src, labels, val, ignore_index);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void total_sum_accum_async<fp32_t>(Scalar alpha,
        const Tensor<fp32_t> &logsumexp, const Tensor<fp32_t> &src,
        const Tensor<int64_t> &class_labels, const Tensor<fp32_t> &val,
        Index ignore_index);

template
void total_sum_accum_async<fp32_fast_tf32_t>(Scalar alpha,
        const Tensor<fp32_fast_tf32_t> &logsumexp,
        const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<int64_t> &class_labels, const Tensor<fp32_t> &val,
        Index ignore_index);

template
void total_sum_accum_async<fp32_fast_fp16_t>(Scalar alpha,
        const Tensor<fp32_fast_fp16_t> &logsumexp,
        const Tensor<fp32_fast_fp16_t> &src,
        const Tensor<int64_t> &class_labels, const Tensor<fp32_t> &val,
        Index ignore_index);

template
void total_sum_accum_async<fp32_fast_bf16_t>(Scalar alpha,
        const Tensor<fp32_fast_bf16_t> &logsumexp,
        const Tensor<fp32_fast_bf16_t> &src,
        const Tensor<int64_t> &class_labels, const Tensor<fp32_t> &val,
        Index ignore_index);

template
void total_sum_accum_async<fp64_t>(Scalar alpha,
        const Tensor<fp64_t> &logsumexp, const Tensor<fp64_t> &src,
        const Tensor<int64_t> &class_labels, const Tensor<fp32_t> &val,
        Index ignore_index);

template
void total_sum_accum_async<bf16_t>(Scalar alpha,
        const Tensor<bf16_t> &logsumexp, const Tensor<bf16_t> &src,
        const Tensor<int64_t> &class_labels, const Tensor<fp32_t> &val,
        Index ignore_index);

template
void total_sum_accum_async<fp16_t>(Scalar alpha, const Tensor<fp16_t> &logsumexp,
        const Tensor<fp16_t> &src, const Tensor<int64_t> &class_labels,
        const Tensor<fp32_t> &val, Index ignore_index);

// Explicit instantiation
template
void total_sum_accum<fp32_t>(Scalar alpha, const Tensor<fp32_t> &logsumexp,
        const Tensor<fp32_t> &src, const Tensor<int64_t> &class_labels,
        const Tensor<fp32_t> &val, Index ignore_index);

template
void total_sum_accum<fp32_fast_tf32_t>(Scalar alpha,
        const Tensor<fp32_fast_tf32_t> &logsumexp,
        const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<int64_t> &class_labels, const Tensor<fp32_t> &val,
        Index ignore_index);

template
void total_sum_accum<fp32_fast_fp16_t>(Scalar alpha,
        const Tensor<fp32_fast_fp16_t> &logsumexp,
        const Tensor<fp32_fast_fp16_t> &src,
        const Tensor<int64_t> &class_labels, const Tensor<fp32_t> &val,
        Index ignore_index);

template
void total_sum_accum<fp32_fast_bf16_t>(Scalar alpha,
        const Tensor<fp32_fast_bf16_t> &logsumexp,
        const Tensor<fp32_fast_bf16_t> &src,
        const Tensor<int64_t> &class_labels, const Tensor<fp32_t> &val,
        Index ignore_index);

template
void total_sum_accum<fp64_t>(Scalar alpha, const Tensor<fp64_t> &logsumexp,
        const Tensor<fp64_t> &src, const Tensor<int64_t> &class_labels,
        const Tensor<fp32_t> &val, Index ignore_index);

template
void total_sum_accum<bf16_t>(Scalar alpha, const Tensor<bf16_t> &logsumexp,
        const Tensor<bf16_t> &src, const Tensor<int64_t> &class_labels,
        const Tensor<fp32_t> &val, Index ignore_index);

template
void total_sum_accum<fp16_t>(Scalar alpha, const Tensor<fp16_t> &logsumexp,
        const Tensor<fp16_t> &src, const Tensor<int64_t> &class_labels,
        const Tensor<fp32_t> &val, Index ignore_index);

} // namespace nntile::tensor
