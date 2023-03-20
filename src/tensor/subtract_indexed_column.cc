/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/subtract_indexed_column.cc
 * Subtraction of value from the indexed column in Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-18
 * */

#include "nntile/tensor/subtract_indexed_column.hh"
#include "nntile/starpu/subtract_indexed_column.hh"

namespace nntile
{
namespace tensor
{

//! Compute total_sum_accum
template<typename T>
void subtract_indexed_column_async(T val,
                                   const Tensor<Index> &class_labels,
                                   const Tensor<T> &dst)
{
    if(class_labels.shape[0] != dst.shape[0])
    {
        throw std::runtime_error("class_labels.shape[0] != dst.shape[0]");
    }
    if(class_labels.basetile_shape[0] != dst.basetile_shape[0])
    {
        throw std::runtime_error("class_labels.basetile_shape[0] != dst.basetile_shape[0]");
    }
    if(dst.shape[1] != dst.basetile_shape[1])
    {
        throw std::runtime_error("dst.shape[1] != dst.basetile_shape[1]");
    }

    // Do actual calculations
    int mpi_rank = starpu_mpi_world_rank();
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        auto class_labels_tile_handle = class_labels.get_tile_handle(i);
        auto class_labels_traits = class_labels.get_tile_traits(i);
        auto dst_tile_handle = dst.get_tile_handle(i);

        int dst_tile_rank = dst_tile_handle.mpi_get_rank();

        // Transfer data
        class_labels_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);

        // Execute on destination node
        if (mpi_rank == dst_tile_rank)
        {
            // Insert task
            starpu::subtract_indexed_column::submit<T>(class_labels_traits.nelems, val, 
                                               class_labels_tile_handle, dst_tile_handle);
        }
        dst_tile_handle.mpi_flush();
    }
}


template<typename T>
void subtract_indexed_column(T val,
                             const Tensor<Index> &class_labels,
                             const Tensor<T> &dst)
{
    subtract_indexed_column_async<T>(val, class_labels, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void subtract_indexed_column_async<fp32_t>(fp32_t val,
                                   const Tensor<Index> &class_labels,
                                   const Tensor<fp32_t> &dst);

template
void subtract_indexed_column_async<fp64_t>(fp64_t val,
                                           const Tensor<Index> &class_labels,
                                           const Tensor<fp64_t> &dst);

// Explicit instantiation
template
void subtract_indexed_column<fp32_t>(fp32_t val,
                                     const Tensor<Index> &class_labels,
                                     const Tensor<fp32_t> &dst);

template
void subtract_indexed_column<fp64_t>(fp64_t val,
                                     const Tensor<Index> &class_labels,
                                     const Tensor<fp64_t> &dst);

} // namespace tensor
} // namespace nntile
