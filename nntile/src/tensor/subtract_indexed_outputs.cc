/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/subtract_indexed_outputs.cc
 * Subtraction of value from certain elements in Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/subtract_indexed_outputs.hh"
#include "nntile/tile/subtract_indexed_outputs.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

template<typename T>
void subtract_indexed_outputs_async(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<T> &dst, Index ignore_index)
{
    if(labels.ndim != dst.ndim-1)
    {
        throw std::runtime_error("labels.ndim != dst.ndim-1");
    }
    for(Index i = 0; i < labels.ndim; ++i)
    {
        if(labels.shape[i] != dst.shape[i+1])
        {
            throw std::runtime_error("labels.shape[i] != dst.shape[i+1]");
        }
        if(labels.basetile_shape[i] != dst.basetile_shape[i+1])
        {
            throw std::runtime_error("labels.basetile_shape[i] != "
                    "dst.basetile_shape[i+1]");
        }
    }
    if(dst.shape[0] != dst.basetile_shape[0])
    {
        throw std::runtime_error("dst.shape[0] != dst.basetile_shape[0]");
    }
    // Do actual calculations
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        auto dst_tile_handle = dst.get_tile_handle(i);
        auto labels_tile = labels.get_tile(i);
        auto dst_tile = dst.get_tile(i);
        tile::subtract_indexed_outputs_async<T>(val, labels_tile, dst_tile,
                ignore_index);
        dst_tile_handle.mpi_flush();
    }
}

template<typename T>
void subtract_indexed_outputs(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<T> &dst, Index ignore_index)
{
    subtract_indexed_outputs_async<T>(val, labels, dst, ignore_index);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void subtract_indexed_outputs_async<fp32_t>(Scalar val,
        const Tensor<int64_t> &labels, const Tensor<fp32_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs_async<fp32_fast_tf32_t>(Scalar val,
        const Tensor<int64_t> &labels, const Tensor<fp32_fast_tf32_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs_async<fp32_fast_fp16_t>(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<fp32_fast_fp16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs_async<fp32_fast_bf16_t>(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<fp32_fast_bf16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs_async<fp64_t>(Scalar val,
        const Tensor<int64_t> &labels, const Tensor<fp64_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs_async<bf16_t>(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<bf16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs_async<fp16_t>(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<fp16_t> &dst, Index ignore_index);

// Explicit instantiation
template
void subtract_indexed_outputs<fp32_t>(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<fp32_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp32_fast_tf32_t>(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<fp32_fast_tf32_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp32_fast_fp16_t>(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<fp32_fast_fp16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp32_fast_bf16_t>(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<fp32_fast_bf16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp64_t>(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<fp64_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<bf16_t>(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<bf16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp16_t>(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<fp16_t> &dst, Index ignore_index);

} // namespace nntile::tensor
