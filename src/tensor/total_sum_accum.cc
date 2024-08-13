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
#include "nntile/starpu/total_sum_accum.hh"
#include "nntile/starpu/clear.hh"

namespace nntile::tensor
{

//! Compute total_sum_accum
template<typename T>
void total_sum_accum_async(Scalar alpha, const Tensor<T> &logsumexp,
        const Tensor<T> &src, const Tensor<int64_t> &labels,
        const Tensor<fp32_t> &val)
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
    int mpi_rank = starpu_mpi_world_rank();
    auto val_tile_handle = val.get_tile_handle(0);
    int val_tile_rank = val_tile_handle.mpi_get_rank();
    for(Index i = 0; i < labels.grid.nelems; ++i)
    {
        // Clean up destination tile on dest node
        auto logsumexp_tile_handle = logsumexp.get_tile_handle(i);
        auto logsumexp_tile_traits = logsumexp.get_tile_traits(i);
        auto labels_tile_handle = labels.get_tile_handle(i);
        auto src_tile_handle = src.get_tile_handle(i);
        // Transfer data to exec_rank=val_tile_rank
        src_tile_handle.mpi_transfer(val_tile_rank, mpi_rank);
        logsumexp_tile_handle.mpi_transfer(val_tile_rank, mpi_rank);
        labels_tile_handle.mpi_transfer(val_tile_rank, mpi_rank);
        // Execute on destination node
        if(mpi_rank == val_tile_rank)
        {
            // Insert task
            starpu::total_sum_accum::submit<T>(alpha, src.shape[0],
                    logsumexp_tile_traits.nelems, logsumexp_tile_handle,
                    src_tile_handle, labels_tile_handle, val_tile_handle);
        }
    }
    val_tile_handle.mpi_flush();
}

template<typename T>
void total_sum_accum(Scalar alpha, const Tensor<T> &logsumexp,
        const Tensor<T> &src, const Tensor<int64_t> &labels,
        const Tensor<fp32_t> &val)
{
    total_sum_accum_async<T>(alpha, logsumexp, src, labels, val);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void total_sum_accum_async<fp32_t>(Scalar alpha,
        const Tensor<fp32_t> &logsumexp, const Tensor<fp32_t> &src,
        const Tensor<int64_t> &class_labels, const Tensor<fp32_t> &val);

template
void total_sum_accum_async<fp32_fast_tf32_t>(Scalar alpha,
        const Tensor<fp32_fast_tf32_t> &logsumexp,
        const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<int64_t> &class_labels, const Tensor<fp32_t> &val);

template
void total_sum_accum_async<fp64_t>(Scalar alpha,
        const Tensor<fp64_t> &logsumexp, const Tensor<fp64_t> &src,
        const Tensor<int64_t> &class_labels, const Tensor<fp32_t> &val);

template
void total_sum_accum_async<bf16_t>(Scalar alpha,
        const Tensor<bf16_t> &logsumexp, const Tensor<bf16_t> &src,
        const Tensor<int64_t> &class_labels, const Tensor<fp32_t> &val);

// Explicit instantiation
template
void total_sum_accum<fp32_t>(Scalar alpha, const Tensor<fp32_t> &logsumexp,
        const Tensor<fp32_t> &src, const Tensor<int64_t> &class_labels,
        const Tensor<fp32_t> &val);

template
void total_sum_accum<fp32_fast_tf32_t>(Scalar alpha,
        const Tensor<fp32_fast_tf32_t> &logsumexp,
        const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<int64_t> &class_labels, const Tensor<fp32_t> &val);

template
void total_sum_accum<fp64_t>(Scalar alpha, const Tensor<fp64_t> &logsumexp,
        const Tensor<fp64_t> &src, const Tensor<int64_t> &class_labels,
        const Tensor<fp32_t> &val);

template
void total_sum_accum<bf16_t>(Scalar alpha, const Tensor<bf16_t> &logsumexp,
        const Tensor<bf16_t> &src, const Tensor<int64_t> &class_labels,
        const Tensor<fp32_t> &val);

} // namespace nntile::tensor
