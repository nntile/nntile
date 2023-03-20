/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/total_sum_accum.cc
 * Total sum accumulating of Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-16
 * */

#include "nntile/tensor/total_sum_accum.hh"
#include "nntile/starpu/total_sum_accum.hh"
#include "nntile/starpu/clear.hh"

namespace nntile
{
namespace tensor
{

//! Compute total_sum_accum
template<typename T>
void total_sum_accum_async(const Tensor<T> &logsumexp,
                           const Tensor<T> &src, const Tensor<Index> &class_labels,
                           const Tensor<T> &val)
{
    // Check dimensions
    if(logsumexp.ndim != class_labels.ndim)
    {
        throw std::runtime_error("logsumexp.ndim != class_labels.ndim");
    }
    // Check shapes of src and dst
    if(logsumexp.ndim != 1)
    {
        throw std::runtime_error("logsumexp.ndim != 1");
    }
    if(val.ndim != 0)
    {
        throw std::runtime_error("val.ndim != 0");
    }
    if(logsumexp.shape[0] != class_labels.shape[0])
    {
        throw std::runtime_error("logsumexp.shape[0] != class_labels.shape[0]");
    }
    if(class_labels.shape[0] != src.shape[0])
    {
        throw std::runtime_error("class_labels.shape[0] != src.shape[0]");
    }
    if(src.basetile_shape[0] != logsumexp.basetile_shape[0])
    {
        throw std::runtime_error("src.basetile_shape[0] != logsumexp.basetile_shape[0]");
    }
    if(src.basetile_shape[0] != class_labels.basetile_shape[0])
    {
        throw std::runtime_error("src.basetile_shape[0] != class_labels.basetile_shape[0]");
    }
    if(src.basetile_shape[1] != src.shape[1])
    {
        throw std::runtime_error("src.basetile_shape[1] != src.shape[1]");
    }
    // Do actual calculations
    int mpi_rank = starpu_mpi_world_rank();
    auto val_tile_handle = val.get_tile_handle(0);
    int val_tile_rank = val_tile_handle.mpi_get_rank();
    if(mpi_rank == val_tile_rank)
    {
        starpu::clear::submit(val_tile_handle);
    }
    for(Index i = 0; i < class_labels.grid.nelems; ++i)
    {
        // Clean up destination tile on dest node
        auto logsumexp_tile_handle = logsumexp.get_tile_handle(i);
        auto logsumexp_tile_traits = logsumexp.get_tile_traits(i);
        auto class_labels_tile_handle = class_labels.get_tile_handle(i);
        auto src_tile_handle = src.get_tile_handle(i);
        // Transfer data to node for src_tile since it is the largest anong others
        src_tile_handle.mpi_transfer(val_tile_rank, mpi_rank);
        logsumexp_tile_handle.mpi_transfer(val_tile_rank, mpi_rank);
        class_labels_tile_handle.mpi_transfer(val_tile_rank, mpi_rank);

        // Execute on destination node
        if(mpi_rank == val_tile_rank)
        {
            // Insert task
            starpu::total_sum_accum::submit<T>(logsumexp_tile_traits.nelems, logsumexp_tile_handle, 
                                               src_tile_handle, class_labels_tile_handle, val_tile_handle);
        }
    }
    val_tile_handle.mpi_flush();
}


template<typename T>
void total_sum_accum(const Tensor<T> &logsumexp,
                     const Tensor<T> &src, const Tensor<Index> &class_labels,
                     const Tensor<T> &val)
{
    total_sum_accum_async<T>(logsumexp, src, class_labels, val);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void total_sum_accum_async<fp32_t>(const Tensor<fp32_t> &logsumexp,
                                   const Tensor<fp32_t> &src, const Tensor<Index> &class_labels,
                                   const Tensor<fp32_t> &val);

template
void total_sum_accum_async<fp64_t>(const Tensor<fp64_t> &logsumexp,
                                   const Tensor<fp64_t> &src, const Tensor<Index> &class_labels,
                                   const Tensor<fp64_t> &val);

// Explicit instantiation
template
void total_sum_accum<fp32_t>(const Tensor<fp32_t> &logsumexp,
                             const Tensor<fp32_t> &src, const Tensor<Index> &class_labels,
                             const Tensor<fp32_t> &val);

template
void total_sum_accum<fp64_t>(const Tensor<fp64_t> &logsumexp,
                             const Tensor<fp64_t> &src, const Tensor<Index> &class_labels,
                             const Tensor<fp64_t> &val);

} // namespace tensor
} // namespace nntile
