/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/subtract_indexed_outputs.cc
 *
 * @version 1.1.0
 * */

#include "nntile/core/subtract_indexed_outputs.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/subtract_indexed_outputs.hh"
#else
#include "nntile/starpu/subtract_indexed_outputs.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

template<typename T>
void subtract_indexed_outputs_async(int starpu_worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<T> &dst, Index ignore_index)
{
    (void)starpu_worker_hint;
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
    #ifdef NNTILE_USE_NNHAUL
    int mpi_rank = 0;
    #else
    int mpi_rank = starpu_mpi_world_rank();
    #endif
    #ifdef NNTILE_USE_NNHAUL
    int dst_rank = 0;
    #else
    int dst_rank = dst.mpi_get_rank();
    #endif
    #ifndef NNTILE_USE_NNHAUL
    labels.mpi_transfer(dst_rank, mpi_rank);
    #endif
    if(mpi_rank == dst_rank)
    {
        const Index n_labels = dst.shape[dst.ndim - 1];
        #ifdef NNTILE_USE_NNHAUL
        haul::subtract_indexed_outputs.submit<std::tuple<T>>(
                n_labels, labels.nelems, ignore_index, val, labels, dst);
        #else
        starpu::subtract_indexed_outputs.submit<std::tuple<T>>(
                n_labels, labels.nelems, ignore_index, val, labels, dst);
        #endif

    }
}

template<typename T>
void subtract_indexed_outputs(int starpu_worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<T> &dst, Index ignore_index)
{
    subtract_indexed_outputs_async<T>(starpu_worker_hint, val, labels, dst, ignore_index);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation
template
void subtract_indexed_outputs_async<fp32_t>(int starpu_worker_hint, Scalar val,
        const Tile<int64_t> &labels, const Tile<fp32_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs_async<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar val,
        const Tile<int64_t> &labels, const Tile<fp32_fast_tf32_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs_async<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_fast_fp16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs_async<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_fast_bf16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs_async<fp64_t>(int starpu_worker_hint, Scalar val,
        const Tile<int64_t> &labels, const Tile<fp64_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs_async<bf16_t>(int starpu_worker_hint, Scalar val,
        const Tile<int64_t> &labels, const Tile<bf16_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs_async<fp16_t>(int starpu_worker_hint, Scalar val,
        const Tile<int64_t> &labels, const Tile<fp16_t> &dst,
        Index ignore_index);

// Explicit instantiation
template
void subtract_indexed_outputs<fp32_t>(int starpu_worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_fast_tf32_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_fast_fp16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_fast_bf16_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<fp64_t>(int starpu_worker_hint, Scalar val, const Tile<int64_t> &labels,
        const Tile<fp64_t> &dst, Index ignore_index);

template
void subtract_indexed_outputs<bf16_t>(int starpu_worker_hint, Scalar val,
        const Tile<int64_t> &labels, const Tile<bf16_t> &dst,
        Index ignore_index);

template
void subtract_indexed_outputs<fp16_t>(int starpu_worker_hint, Scalar val,
        const Tile<int64_t> &labels, const Tile<fp16_t> &dst,
        Index ignore_index);

} // namespace nntile::core
