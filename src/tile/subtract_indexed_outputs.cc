/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/subtract_indexed_outputs.cc
 *
 * @version 1.1.0
 * */

#include "nntile/tile/subtract_indexed_outputs.hh"
#include "nntile/starpu/subtract_indexed_outputs.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

template<typename T>
void subtract_indexed_outputs_async(Scalar val, const Tile<int64_t> &labels,
        const Tile<T> &dst, Index ignore_index)
{
// TODO - add description
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
    }
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    labels.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        // Insert task
        starpu::subtract_indexed_outputs.submit<std::tuple<T>>(dst.shape[0],
                labels.nelems, ignore_index, val, labels, dst);
    }
}

template<typename T>
void subtract_indexed_outputs(Scalar val, const Tile<int64_t> &labels,
        const Tile<T> &dst, Index ignore_index)
{
    subtract_indexed_outputs_async<T>(val, labels, dst, ignore_index);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void subtract_indexed_outputs_async<fp32_t>(Scalar val,
        const Tile<int64_t> &labels, const Tile<fp32_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs_async<fp32_fast_tf32_t>(Scalar val,
        const Tile<int64_t> &labels, const Tile<fp32_fast_tf32_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs_async<fp32_fast_fp16_t>(Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_fast_fp16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs_async<fp32_fast_bf16_t>(Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_fast_bf16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs_async<fp64_t>(Scalar val,
        const Tile<int64_t> &labels, const Tile<fp64_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs_async<bf16_t>(Scalar val,
        const Tile<int64_t> &labels, const Tile<bf16_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs_async<fp16_t>(Scalar val,
        const Tile<int64_t> &labels, const Tile<fp16_t> &dst,
        Index ignore_index);

// Explicit instantiation
template
void subtract_indexed_outputs<fp32_t>(Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp32_fast_tf32_t>(Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_fast_tf32_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp32_fast_fp16_t>(Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_fast_fp16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp32_fast_bf16_t>(Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_fast_bf16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp64_t>(Scalar val, const Tile<int64_t> &labels,
        const Tile<fp64_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<bf16_t>(Scalar val,
        const Tile<int64_t> &labels, const Tile<bf16_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs<fp16_t>(Scalar val,
        const Tile<int64_t> &labels, const Tile<fp16_t> &dst,
        Index ignore_index);

} // namespace nntile::tile
